from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, TypeAlias

import torch

from cezo_fl.client import AbstractClient, LocalUpdateResult
from cezo_fl.util.metrics import Metric


def parallalizable_client_job(
    client: AbstractClient,
    pull_seeds_list: Sequence[Sequence[int]],
    pull_grad_list: Sequence[Sequence[torch.Tensor]],
    local_update_seeds: Sequence[int],
    server_device: torch.device,
) -> LocalUpdateResult:
    """
    Run client pull and local update in parallel.
    This function is added to make better use of multi-gpu set up.
    Each client can be deployed to a separate gpu. Thus we can run all clients in parallel.

    Note:
    This function also make sure the data passed to/from client are converted to correct device.
    We should only do cross device operation here
    """
    # need no_grad because the outer-most no_grad context manager does not affect
    # operation inside sub-thread
    with torch.no_grad():
        # step 1: map pull_grad_list data to client's device
        transfered_grad_list = [
            [tensor.to(client.device) for tensor in tensors] for tensors in pull_grad_list
        ]

        # step 2: client pull to update its model to latest
        client.pull_model(pull_seeds_list, transfered_grad_list)

        # step 3: client local update and get its result
        client_local_update_result = client.local_update(seeds=local_update_seeds)

    # move result to server device and return
    return client_local_update_result.to(server_device)


# Outer list is for clients and inner list for local update data
LOCAL_GRAD_SCALAR_LIST: TypeAlias = list[list[torch.Tensor]]


def execute_sampled_clients(
    server,  # TODO: add good typehint here
    sampled_client_index: Sequence[int],
    seeds: Sequence[int],
    *,
    parallel: bool = False,
) -> tuple[Metric, Metric, LOCAL_GRAD_SCALAR_LIST]:
    local_grad_scalar_list: LOCAL_GRAD_SCALAR_LIST = []  # Clients X Local_update
    step_train_loss = Metric("Step train loss")
    step_train_accuracy = Metric("Step train accuracy")

    if parallel:
        with ThreadPoolExecutor() as executor:
            futures = []
            for index in sampled_client_index:
                client = server.clients[index]
                last_update_iter = server.client_last_updates[index]
                # The seed and grad in last_update_iter is fetched as well
                # Note at that iteration, we just reset the client model so that iteration
                # information is needed as well.
                seeds_list = server.seed_grad_records.fetch_seed_records(last_update_iter)
                grad_list = server.seed_grad_records.fetch_grad_records(last_update_iter)

                futures.append(
                    executor.submit(
                        parallalizable_client_job,
                        client,
                        seeds_list,
                        grad_list,
                        seeds,
                        server.device,
                    )
                )

            client_results = [f.result(timeout=1e4) for f in futures]
    else:
        client_results = []
        for index in sampled_client_index:
            client = server.clients[index]
            last_update_iter = server.client_last_updates[index]
            # The seed and grad in last_update_iter is fetched as well
            # Note at that iteration, we just reset the client model so that iteration
            # information is needed as well.
            seeds_list = server.seed_grad_records.fetch_seed_records(last_update_iter)
            grad_list = server.seed_grad_records.fetch_grad_records(last_update_iter)
            client_results.append(
                parallalizable_client_job(client, seeds_list, grad_list, seeds, server.device)
            )

    for client_local_update_result in client_results:
        step_train_loss.update(client_local_update_result.step_loss)
        step_train_accuracy.update(client_local_update_result.step_accuracy)
        local_grad_scalar_list.append(client_local_update_result.grad_tensors)

    return step_train_loss, step_train_accuracy, local_grad_scalar_list
