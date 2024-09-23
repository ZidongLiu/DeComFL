from huggingface_hub.repository import atexit
import torch
import grpc
from cezo_grpc import sample_pb2
from cezo_grpc import sample_pb2_grpc
from cezo_grpc import data_helper

from cezo_fl import fl_helpers
from cezo_fl import client
import config
import preprocess
import decomfl_main
import time


def setup_client(args, client_index):
    device_map, train_loaders, _ = preprocess.preprocess(args)
    client_name = fl_helpers.get_client_name(client_index)
    client_device = device_map[client_name]
    (
        client_model,
        client_criterion,
        client_optimizer,
        client_grad_estimator,
        client_accuracy_func,
    ) = decomfl_main.prepare_settings_underseed(args, client_device)
    client_model.to(client_device)

    return client.ResetClient(
        client_model,
        train_loaders[client_index],
        client_grad_estimator,
        client_optimizer,
        client_criterion,
        client_accuracy_func,
        client_device,
    )


def repeat_every(fn, pass_fn, repeat_interval=1):
    while True:
        response = fn()
        if pass_fn(response):
            return response
        time.sleep(repeat_interval)


def get_stub():
    rpc_master_addr = "localhost"
    rpc_master_port = 4242
    channel = grpc.insecure_channel(f"{rpc_master_addr}:{rpc_master_port}")
    ps_stub = sample_pb2_grpc.SampleServerStub(channel)
    return ps_stub


def train_with_args(args):
    ps_stub = get_stub()

    connect_result = repeat_every(
        lambda: ps_stub.Connect(sample_pb2.EmptyRequest()), lambda x: x.successful
    )
    client_index = connect_result.clientIndex
    print(f"connected as client: {client_index}")
    # when program exits, we need to disconnect this client from server
    atexit.register(
        lambda: ps_stub.Disconnect(sample_pb2.DisconnectRequest(clientIndex=client_index))
    )

    with torch.no_grad():
        client_instance = setup_client(args, client_index)

        def try_to_join_iteration():
            join_result = repeat_every(
                lambda: ps_stub.TryToJoinIteration(
                    sample_pb2.TryToJoinIterationRequest(clientIndex=client_index)
                ),
                lambda x: x.successful,
            )

            print("join iteration")
            pull_seeds_list = data_helper.protobuf_to_py_list_of_list_of_ints(join_result.pullSeeds)
            raw_grad_list = data_helper.protobuf_to_py_list_of_list_of_list_of_floats(
                join_result.pullGrads
            )
            tensor_grad_list = [
                [torch.tensor(v, device=client_instance.device) for v in vv] for vv in raw_grad_list
            ]
            iteration_seeds = data_helper.protobuf_to_py_list_of_ints(join_result.iterationSeeds)

            # step 2: client pull to update its model to latest
            client_instance.pull_model(pull_seeds_list, tensor_grad_list)

            # step 3: client local update and get its result
            client_local_update_result = client_instance.local_update(seeds=iteration_seeds)

            print("submit result")
            ps_stub.SubmitIteration(
                sample_pb2.SubmitIterationRequest(
                    clientIndex=client_index,
                    gradTensors=data_helper.py_to_protobuf_list_of_list_of_floats(
                        [t.tolist() for t in client_local_update_result.grad_tensors]
                    ),
                    stepAccuracy=client_local_update_result.step_accuracy,
                    stepLoss=client_local_update_result.step_loss,
                )
            )

        repeat_every(try_to_join_iteration, lambda x: False)


if __name__ == "__main__":
    args = config.get_params_grpc().parse_args()
    if args.dataset == "shakespeare":
        args.num_clients = 139
    print(args)
    train_with_args(args)
