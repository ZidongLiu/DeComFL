import functools
from os import path
import torch


from tensorboardX import SummaryWriter
from tqdm import tqdm

from byzantine import aggregation as byz_agg


from byzantine import attack as byz_attack
from cezo_fl.client import ResetClient
from cezo_fl.fl_helpers import get_client_name, get_server_name
from cezo_fl.server import CeZO_Server
from cezo_fl.util import model_helpers
from experiment_helper import prepare_settings
from experiment_helper.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    FederatedLearningSetting,
    RGESetting,
    ByzantineSetting,
)
from experiment_helper.device import use_device
from experiment_helper.data import get_dataloaders
import os
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import time
import os


class CliSetting(
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    FederatedLearningSetting,
    RGESetting,
    ByzantineSetting,
):
    """
    This is a replacement for regular argparse module.
    We used a third party library pydantic_setting to make command line interface easier to manage.
    Example:
    if __name__ == "__main__":
        args = CliSetting()

    args will have all parameters defined by all components.
    """

    pass


def setup_server_and_clients(
    args: CliSetting, device_map: dict[str, torch.device], train_loaders
) -> CeZO_Server:
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )
    client_rrefs = []

    for i in range(args.num_clients):
        client_ref = rpc.remote(
            to=get_client_name(i), func=rpc_setup_client_from_server, args=(i, args)
        )
        client_rrefs += [client_ref]

    server_device = device_map[get_server_name()]
    server = CeZO_Server(
        client_rrefs,
        server_device,
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_update_steps,
    )

    # set server tools
    server_model = prepare_settings.get_model(
        dataset=args.dataset, model_setting=args.model_setting, seed=args.seed
    ).to(server_device)
    server_optimizer = prepare_settings.get_optimizer(
        model=server_model, dataset=args.dataset, optimizer_setting=args.optimizer_setting
    )
    server_grad_estimator = prepare_settings.get_random_gradient_estimator(
        model=server_model,
        device=server_device,
        rge_setting=args.rge_setting,
        model_setting=args.model_setting,
    )

    server.set_server_model_and_criterion(
        server_model,
        model_inferences.test_inference,
        metrics.test_loss,
        metrics.test_acc,
        server_optimizer,
        server_grad_estimator,
    )

    return server


def rpc_setup_client_from_server(rank, args):
    device_map = use_device(args.device_setting, args.num_clients)
    train_loaders, _ = get_dataloaders(
        args.data_setting, args.num_clients, args.seed, args.get_hf_model_name()
    )

    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )

    client_name = get_client_name(rank)
    client_device = device_map[client_name]
    client_model = prepare_settings.get_model(
        dataset=args.dataset, model_setting=args.model_setting, seed=args.seed
    ).to(client_device)
    client_optimizer = prepare_settings.get_optimizer(
        model=client_model, dataset=args.dataset, optimizer_setting=args.optimizer_setting
    )
    client_grad_estimator = prepare_settings.get_random_gradient_estimator(
        model=client_model,
        device=client_device,
        rge_setting=args.rge_setting,
        model_setting=args.model_setting,
    )

    client = ResetClient(
        client_model,
        model_inferences.train_inference,
        train_loaders[rank],
        client_grad_estimator,
        client_optimizer,
        metrics.train_loss,
        metrics.train_acc,
        client_device,
    )
    return client


def rpc_worker(rank, world_size, args: CliSetting):
    # rank >= 1 for worker, thus need to minus to keep it the same as device_map's name
    rpc.init_rpc(get_client_name(rank - 1), rank=rank, world_size=world_size)
    rpc.shutdown()


def print_work_info(client_ref):
    print("worker", rpc.get_worker_info())
    print("client", client_ref.owner_name())
    print(client_ref.to_here().device)
    return client_ref.to_here().device


def rpc_server(rank, world_size, args: CliSetting):
    """Caller function that sends an RPC to the worker."""
    rpc.init_rpc(get_server_name(), rank=rank, world_size=world_size)
    print(args)
    # Name of the worker to send the task to
    device_map = use_device(args.device_setting, args.num_clients)
    _, test_loader = get_dataloaders(
        args.data_setting, args.num_clients, args.seed, args.get_hf_model_name()
    )
    server = setup_server_and_clients(args, device_map, test_loader)
    # add 5s sleep here, need to fix this
    # time.sleep(5)

    # ret = rpc.rpc_async(
    #     get_client_name(0),
    #     print_work_info,
    #     (server.clients[0],),
    # )
    # print(ret.wait())
    # server.train_one_step(0)
    with tqdm(total=args.iterations, desc="Training:") as t, torch.no_grad():
        for ite in range(args.iterations):
            step_loss, step_accuracy = server.train_one_step(ite)
            t.set_postfix({"Loss": step_loss, "Accuracy": step_accuracy})
            t.update(1)

            # eval
            if args.eval_iterations != 0 and (ite + 1) % args.eval_iterations == 0:
                eval_loss, eval_accuracy = server.eval_model(test_loader)

    rpc.shutdown()


def fn(rank, world_size, args):
    if rank == 0:
        return rpc_server(rank, world_size, args)
    else:
        return rpc_worker(rank, world_size, args)


if __name__ == "__main__":
    args = CliSetting()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    world_size = args.num_clients + 1  # We will have 1 server + num_clients worker

    # Spawn two processes, one for the caller and one for the worker
    mp.spawn(fn, args=(world_size, args), nprocs=world_size, join=True)
