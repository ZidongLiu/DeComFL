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


class ServerStatus(Enum):
    connecting = "connecting"
    training = "training"
    aggregating = "aggregating"


def find_first(lst: list, check_fn) -> int:
    try:
        return next(i for i, v in enumerate(lst) if check_fn(v))
    except StopIteration:
        return -1


class SampleServer(sample_pb2_grpc.SampleServerServicer):
    def __init__(self):
        self.num_clients = 3
        self.num_sample_clients = 2
        self.local_update_steps = 1
        self.seed_grad_records = server.SeedAndGradientRecords()
        self.client_last_updates = [0 for _ in range(self.num_clients)]

        self.status = ServerStatus.connecting

        self.connect_lock = threading.Lock()
        self.connected_clients: list[bool] = [False for _ in range(self.num_clients)]

        self.iteration: int = -1
        self.iteration_seeds: list[int] = []
        self.iteration_sampled_clients: list[int] = []
        self.iteration_local_grad_scalar: dict[int, list[torch.Tensor]] = {}

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

    def try_swtich_from_connecting_to_training(self):
        # 1. check for current status == connecting
        if self.status is not ServerStatus.connecting:
            return
        # 2. check for connected_clients vs num_clients
        if not all(self.connected_clients):
            return
        # 3. initialize training data for next iteration
        print("swtich from connecting to training")
        self.preprare_for_next_iteration()
        # 4. change status to training
        self.change_status(ServerStatus.training)

    def swtich_to_connecting(self):
        if all(self.connected_clients):
            return
        print("Switch to Connecting")
        self.change_status(ServerStatus.connecting)

    def _has_iteration_finished(self):
        return all(
            [
                self.iteration_local_grad_scalar.get(client_index)
                for client_index in self.iteration_sampled_clients
            ]
        )

    def _get_iteration_grad_scalar_list(self):
        print("getting _get_iteration_grad_scalar_list")
        return [
            self.iteration_local_grad_scalar[client_index]
            for client_index in self.iteration_sampled_clients
        ]

    def _aggregate_and_update_server_record(self):
        local_grad_scalar_list = no_byz(self._get_iteration_grad_scalar_list())
        grad_scalar = mean(self.num_sample_clients, local_grad_scalar_list)

        self.seed_grad_records.add_records(seeds=self.iteration_seeds, grad=grad_scalar)
        # Optional: optimize the memory. Remove is exclusive, i.e., the min last updates
        # information is still kept.
        self.seed_grad_records.remove_too_old(earliest_record_needs=min(self.client_last_updates))

    def try_switch_from_training_to_aggregating(self):
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
        # 5. change to next training
        self.preprare_for_next_iteration()
        self.change_status(ServerStatus.training)

    def Connect(self, request, context):
        if self.status is not ServerStatus.connecting:
            return sample_pb2.ConnectResponse(successful=False, clientIndex=-1)

        self.connect_lock.acquire()
        client_index = self._get_next_connect_client_index()
        if client_index == -1:
            print("All clients slot are engaged, decline this connect request")
            return sample_pb2.ConnectResponse(successful=False, clientIndex=client_index)

        print(f"try to assign {client_index}")
        self.connected_clients[client_index] = True
        self.try_swtich_from_connecting_to_training()
        self.connect_lock.release()
        return sample_pb2.ConnectResponse(successful=True, clientIndex=client_index)

    def Disconnect(self, request, context):
        # this can happen at any status, we force status to be connecting when this method is called
        # TODO: depending on status, need to abort current iteration. May need to
        self.connect_lock.acquire()
        client_index = request.clientIndex
        print(f"client {client_index} disconnected, waiting for new client to connect")
        self.connected_clients[client_index] = False
        # next connected client will need to update from 0's iteration
        self.client_last_updates[client_index] = 0
        self.swtich_to_connecting()
        self.connect_lock.release()

        return sample_pb2.EmptyResponse()

    def TryToJoinIteration(self, request, context):
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

    # def PullGradsAndSeeds(self, request, context):
    #     seeds = sample_pb2.ListOfListOfInts(data=[sample_pb2.ListOfInts(data=[1])])
    #     grads = sample_pb2.ListOfListOfListOfFloats(
    #         data=[sample_pb2.ListOfListOfFloats(data=[sample_pb2.ListOfFloats(data=[0.3])])]
    #     )
    #     return sample_pb2.PullGradsAndSeedsResponse(seeds=seeds, grads=grads)

    def _apply_client_local_update_result(self, client_index, local_update_result):
        if client_index not in self.iteration_sampled_clients:
            return

        if client_index not in self.iteration_local_grad_scalar:
            self.iteration_local_grad_scalar[client_index] = local_update_result

    def SubmitIteration(self, request, context):
        client_index = request.clientIndex
        print(f"submit iteration from {client_index}")

        if (
            self.status is not ServerStatus.training
            or client_index not in self.iteration_sampled_clients
        ):
            return sample_pb2.EmptyResponse()

        self.connect_lock.acquire()
        raw_grad_list = data_helper.protobuf_to_py_list_of_list_of_floats(request.gradTensors)
        grad_tensors = [torch.tensor(v) for v in raw_grad_list]
        self._apply_client_local_update_result(client_index, grad_tensors)
        self.try_switch_from_training_to_aggregating()
        self.connect_lock.release()
        return sample_pb2.EmptyResponse()


def serve(rpc_master_port, rpc_num_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=rpc_num_workers))
    sample_pb2_grpc.add_SampleServerServicer_to_server(SampleServer(), server)
    server.add_insecure_port(f"localhost:{rpc_master_port}")
    print(f"Parameter server starting on [::]:{rpc_master_port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve(4242, 8)
