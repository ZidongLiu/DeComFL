import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import time
import os


def client(rank, world_size):
    """Worker function that runs an RPC server."""
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    # Wait for RPC calls
    rpc.shutdown()


def run_a(a_ref: torch.distributed.rpc.RRef):
    return a_ref.to_here().p()


class A:
    def __init__(self, init_val: int):
        self.init_val = init_val

    def p(self):
        print(self.init_val)
        return torch.tensor(self.init_val)


def remote_task(value):
    """A simple task that the worker will execute."""
    return A(value)


def server(rank, world_size):
    """Caller function that sends an RPC to the worker."""
    rpc.init_rpc(f"caller{rank}", rank=rank, world_size=world_size)

    # Name of the worker to send the task to
    worker_name = "worker1"

    # Send an RPC request to the worker to execute `remote_task`
    a_ref = rpc.remote(worker_name, remote_task, args=(123,))

    ret_future = rpc.rpc_async(worker_name, run_a, args=(a_ref,))

    print(f"server receives {ret_future.wait()}")

    rpc.shutdown()


def fn(rank, world_size):
    if rank == 0:
        return server(rank, world_size)
    else:
        return client(rank, world_size)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2  # We will have 1 caller and 1 worker

    # Spawn two processes, one for the caller and one for the worker
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)
