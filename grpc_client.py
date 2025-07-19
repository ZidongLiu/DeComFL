from huggingface_hub.repository import atexit
import torch
import grpc
import time

from cezo_grpc import sample_pb2
from cezo_grpc import sample_pb2_grpc
from cezo_grpc import data_helper
from cezo_grpc import cli_interface

from cezo_fl import fl_helpers
from cezo_fl import client

from experiment_helper import prepare_settings
from experiment_helper.device import use_device
from experiment_helper.data import get_dataloaders


def setup_client(args: cli_interface.CliSetting, client_index: int):
    device_map = use_device(args.device_setting, args.num_clients)
    train_loaders, _ = get_dataloaders(
        args.data_setting, args.num_clients, args.seed, args.get_hf_model_name()
    )
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )
    client_name = fl_helpers.get_client_name(client_index)
    client_device = device_map[client_name]
    client_model = prepare_settings.get_model(
        dataset=args.dataset, model_setting=args.model_setting, seed=args.seed
    ).to(client_device)
    client_optimizer = prepare_settings.get_optimizer(
        model=client_model, dataset=args.dataset, optimizer_setting=args.optimizer_setting
    )
    client_grad_estimator = prepare_settings.get_gradient_estimator(
        model=client_model,
        device=client_device,
        rge_setting=args.rge_setting,
        model_setting=args.model_setting,
    )

    return client.ResetClient(
        client_model,
        model_inferences.train_inference,
        train_loaders[client_index],
        client_grad_estimator,
        client_optimizer,
        metrics.train_loss,
        metrics.train_acc,
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


def train_with_args(args: cli_interface.CliSetting):
    ps_stub = get_stub()

    connect_result = repeat_every(
        lambda: ps_stub.Connect(sample_pb2.EmptyRequest()),  # type: ignore[attr-defined]
        lambda x: x.successful,
    )
    client_index: int = connect_result.clientIndex
    print(f"connected as client: {client_index}")
    # when program exits, we need to disconnect this client from server
    atexit.register(
        lambda: ps_stub.Disconnect(sample_pb2.DisconnectRequest(clientIndex=client_index))  # type: ignore[attr-defined]
    )

    with torch.no_grad():
        client_instance = setup_client(args, client_index)

        def try_to_join_iteration():
            join_result = repeat_every(
                lambda: ps_stub.TryToJoinIteration(
                    sample_pb2.TryToJoinIterationRequest(clientIndex=client_index)  # type: ignore[attr-defined]
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
                sample_pb2.SubmitIterationRequest(  # type: ignore[attr-defined]
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
    args = cli_interface.CliSetting()
    print(args)
    train_with_args(args)
