import grpc
import threading
from concurrent import futures
import random
from enum import Enum
import torch

from cezo_grpc import data_helper
from cezo_grpc import sample_pb2_grpc
from cezo_grpc import sample_pb2
from cezo_fl import server
from byzantine.aggregation import mean
from byzantine.attack import no_byz
import config


class ServerStatus(Enum):
    connecting = "connecting"
    training = "training"
    aggregating = "aggregating"
    evaluating = "evaluating"
    # TODO: implement finishing logic when training iteration reaches the target
    finishing = "finishing"


def find_first(lst: list, check_fn) -> int:
    try:
        return next(i for i, v in enumerate(lst) if check_fn(v))
    except StopIteration:
        return -1


class SampleServer(sample_pb2_grpc.SampleServerServicer):
    def __init__(self, num_clients, num_sample_clients, local_update_steps):
        self.should_eval = True
        self.eval_iteration = 25

        self.num_clients = num_clients
        self.num_sample_clients = num_sample_clients
        self.local_update_steps = local_update_steps

        self.seed_grad_records = server.SeedAndGradientRecords()
        self.client_last_updates = [0 for _ in range(self.num_clients)]
        self.status = ServerStatus.connecting

        self.lock = threading.Lock()

        self.connected_clients: list[bool] = [False for _ in range(self.num_clients)]

        self.eval_client_connected: bool = False
        self.eval_client_last_update: int = 0

        self.iteration: int = -1
        self.iteration_seeds: list[int] = []
        self.iteration_sampled_clients: list[int] = []
        self.iteration_local_grad_scalar: dict[int, list[torch.Tensor]] = {}

    def _get_connect_status(self):
        if self.should_eval:
            return f"train clients: {self.connected_clients}, eval client: {self.eval_client_connected}"
        else:
            return f"train clients: {self.connected_clients}"

    def _get_next_connect_client_index(self) -> int:
        return find_first(self.connected_clients, lambda x: not x)

    def change_status(self, new_status: Enum) -> None:
        self.status = new_status

    def preprare_for_next_iteration(self):
        print(f"finish iteration {self.iteration}, starting next iteration")
        self.iteration += 1
        self.iteration_seeds = [random.randint(0, 1000000) for _ in range(self.local_update_steps)]
        self.iteration_sampled_clients = random.sample(
            range(self.num_clients), self.num_sample_clients
        )
        self.iteration_local_grad_scalar = {}
        print(f"Iteration: {self.iteration}, sampled_clients: {self.iteration_sampled_clients}")

    def _should_connect(self) -> bool:
        training_not_all_connected = not all(self.connected_clients)
        should_eval_and_eval_not_connected = self.should_eval and not self.eval_client_connected
        return training_not_all_connected or should_eval_and_eval_not_connected

    def try_swtich_from_connecting_to_training(self) -> None:
        # 1. check for current status == connecting
        if self.status is not ServerStatus.connecting:
            return
        # 2. check for connected_clients vs num_clients
        if self._should_connect():
            return
        # 3. initialize training data for next iteration
        print("swtich from connecting to training")
        self.preprare_for_next_iteration()
        # 4. change status to training
        self.change_status(ServerStatus.training)

    def swtich_to_connecting(self) -> None:
        if not self._should_connect():
            return
        print("Switch to Connecting")
        self.change_status(ServerStatus.connecting)

    def _has_iteration_finished(self) -> bool:
        return all(
            [
                self.iteration_local_grad_scalar.get(client_index)
                for client_index in self.iteration_sampled_clients
            ]
        )

    def _get_iteration_grad_scalar_list(self) -> list[list[torch.Tensor]]:
        return [
            self.iteration_local_grad_scalar[client_index]
            for client_index in self.iteration_sampled_clients
        ]

    def _aggregate_and_update_server_record(self) -> None:
        local_grad_scalar_list = no_byz(self._get_iteration_grad_scalar_list())
        grad_scalar = mean(local_grad_scalar_list)

        self.seed_grad_records.add_records(seeds=self.iteration_seeds, grad=grad_scalar)
        # Optional: optimize the memory. Remove is exclusive, i.e., the min last updates
        # information is still kept.
        if self.should_eval:
            last_update_iterations = self.client_last_updates + [self.eval_client_last_update]
        else:
            last_update_iterations = self.client_last_updates

        self.seed_grad_records.remove_too_old(earliest_record_needs=min(last_update_iterations))

    def try_switch_from_training_to_aggregating(self) -> None:
        # 1. check for current status == training
        if self.status is not ServerStatus.training:
            return
        # 2. check if sampled client all have return result
        if not self._has_iteration_finished():
            return
        print("swtich from training to aggregating")
        # 3. change status to aggregating
        self.change_status(ServerStatus.aggregating)
        # 4. update seed_grad_records
        self._aggregate_and_update_server_record()
        # 5. change to next training or evaluating
        if self.should_eval and self.iteration % self.eval_iteration == 0:
            self.switch_from_aggregating_to_evaluating()
        else:
            self.switch_from_aggregating_to_training()

    def switch_from_aggregating_to_training(self) -> None:
        print("swtich from aggregating to training")
        self.preprare_for_next_iteration()
        self.change_status(ServerStatus.training)

    def switch_from_aggregating_to_evaluating(self) -> None:
        print("swtich from aggregating to evaluating")
        self.change_status(ServerStatus.evaluating)

    def switch_from_evaluating_to_training(self) -> None:
        print("swtich from evaluating to training")
        self.preprare_for_next_iteration()
        self.change_status(ServerStatus.training)

    def Connect(self, request, context):
        with self.lock:
            if self.status is not ServerStatus.connecting:
                return sample_pb2.ConnectResponse(successful=False, clientIndex=-1)

            client_index = self._get_next_connect_client_index()
            if client_index == -1:
                print("All clients slot are engaged, decline this connect request")
                return sample_pb2.ConnectResponse(successful=False, clientIndex=client_index)

            self.connected_clients[client_index] = True
            print(f"Just assigned {client_index}. {self._get_connect_status()}")
            self.try_swtich_from_connecting_to_training()
            return sample_pb2.ConnectResponse(successful=True, clientIndex=client_index)

    def Disconnect(self, request, context):
        # this can happen at any status, we force status to be connecting when this method is called
        # TODO: depending on status, need to abort current iteration. May need to
        with self.lock:
            client_index = request.clientIndex
            print(f"client {client_index} disconnected, waiting for new client to connect")
            self.connected_clients[client_index] = False
            # next connected client will need to update from 0's iteration
            self.client_last_updates[client_index] = 0
            self.swtich_to_connecting()

            return sample_pb2.EmptyResponse()

    def TryToJoinIteration(self, request, context):
        with self.lock:
            client_index = request.clientIndex
            if (
                self.status is not ServerStatus.training
                or client_index not in self.iteration_sampled_clients
            ):
                return sample_pb2.TryToJoinIterationResponse(
                    successful=False,
                    pullSeeds=data_helper.py_to_protobuf_list_of_list_of_ints([]),
                    pullGrads=data_helper.py_to_protobuf_list_of_list_of_list_of_floats([]),
                    iterationSeeds=data_helper.py_to_protobuf_list_of_ints([]),
                )

            last_update_iter = self.client_last_updates[client_index]
            # The seed and grad in last_update_iter is fetched as well
            # Note at that iteration, we just reset the client model so that iteration
            # information is needed as well.
            seeds_list = self.seed_grad_records.fetch_seed_records(last_update_iter)
            grad_list = self.seed_grad_records.fetch_grad_records(last_update_iter)

            self.client_last_updates[client_index] = self.iteration

            return sample_pb2.TryToJoinIterationResponse(
                successful=True,
                pullSeeds=data_helper.py_to_protobuf_list_of_list_of_ints(seeds_list),
                pullGrads=data_helper.py_to_protobuf_list_of_list_of_list_of_floats(
                    [[ts.tolist() for ts in vv] for vv in grad_list]
                ),
                iterationSeeds=data_helper.py_to_protobuf_list_of_ints(self.iteration_seeds),
            )

    def _apply_client_local_update_result(self, client_index, local_update_result):
        if client_index not in self.iteration_sampled_clients:
            return

        if client_index not in self.iteration_local_grad_scalar:
            self.iteration_local_grad_scalar[client_index] = local_update_result

    def SubmitIteration(self, request, context):
        with self.lock:
            client_index = request.clientIndex
            print(f"submit iteration from {client_index}")

            if (
                self.status is not ServerStatus.training
                or client_index not in self.iteration_sampled_clients
            ):
                return sample_pb2.EmptyResponse()

            raw_grad_list = data_helper.protobuf_to_py_list_of_list_of_floats(request.gradTensors)
            grad_tensors = [torch.tensor(v) for v in raw_grad_list]
            self._apply_client_local_update_result(client_index, grad_tensors)
            self.try_switch_from_training_to_aggregating()
            return sample_pb2.EmptyResponse()

    def ConnectEval(self, request, context):
        with self.lock:
            if self.status is not ServerStatus.connecting:
                return sample_pb2.ConnectResponse(successful=False, clientIndex=-1)

            if self.eval_client_connected:
                print("A eval client is already connected!")
                return sample_pb2.ConnectResponse(successful=False, clientIndex=-1)

            self.eval_client_connected = True
            print(f"Eval Client connected. {self._get_connect_status()}")
            self.try_swtich_from_connecting_to_training()
            return sample_pb2.ConnectResponse(successful=True, clientIndex=-1)

    def DisconnectEval(self, request, context):
        # this can happen at any status, we force status to be connecting when this method is called
        # TODO: depending on status, need to abort current iteration. May need to
        with self.lock:
            print("Eval client disconnected, waiting for new client to connect")
            self.eval_client_connected = False
            # next connected client will need to update from 0's iteration
            self.eval_client_last_update = 0

            self.swtich_to_connecting()

    def TryToEval(self, request, context):
        with self.lock:
            if self.status is not ServerStatus.evaluating:
                return sample_pb2.TryToJoinIterationResponse(
                    successful=False,
                    pullSeeds=data_helper.py_to_protobuf_list_of_list_of_ints([]),
                    pullGrads=data_helper.py_to_protobuf_list_of_list_of_list_of_floats([]),
                    iterationSeeds=data_helper.py_to_protobuf_list_of_ints([]),
                )

            # The seed and grad in last_update_iter is fetched as well
            # Note at that iteration, we just reset the client model so that iteration
            # information is needed as well.
            seeds_list = self.seed_grad_records.fetch_seed_records(self.eval_client_last_update)
            grad_list = self.seed_grad_records.fetch_grad_records(self.eval_client_last_update)
            self.eval_client_last_update = self.iteration
            return sample_pb2.TryToJoinIterationResponse(
                successful=True,
                pullSeeds=data_helper.py_to_protobuf_list_of_list_of_ints(seeds_list),
                pullGrads=data_helper.py_to_protobuf_list_of_list_of_list_of_floats(
                    [[ts.tolist() for ts in vv] for vv in grad_list]
                ),
                iterationSeeds=data_helper.py_to_protobuf_list_of_ints([]),
            )

    def SubmitEvaluation(self, request, context):
        with self.lock:
            if self.status is not ServerStatus.evaluating:
                return sample_pb2.EmptyResponse()

            eval_loss, eval_accuracy = request.evalLoss, request.evalAccuracy
            print(
                f"\nEvaluation(Iteration {self.iteration}): ",
                f"Eval Loss:{eval_loss:.4f}, " f"Accuracy:{eval_accuracy * 100:.2f}%",
            )
            self.switch_from_evaluating_to_training()

            return sample_pb2.EmptyResponse()


def serve(args):
    rpc_master_port = args.rpc_master_port
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.rpc_num_workers))
    sample_pb2_grpc.add_SampleServerServicer_to_server(
        SampleServer(args.num_clients, args.num_sample_clients, args.local_update_steps), server
    )
    port_str = f"{args.rpc_master_addr}:{rpc_master_port}"
    server.add_insecure_port(port_str)
    print(f"Parameter server starting on {port_str}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    args = config.get_params_grpc().parse_args()
    if args.dataset == "shakespeare":
        args.num_clients = 139
    print(args)
    serve(args)
