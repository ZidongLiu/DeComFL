from multiprocessing import Process

from grpc_client import train_with_args
from grpc_eval_client import eval_with_args
from grpc_server import serve
from cezo_grpc import cli_interface

import time

if __name__ == "__main__":
    args = cli_interface.CliSetting()
    print(args)

    Process(target=serve, args=(args,)).start()

    # HACK: sleep 2 seconds to make sure server can spin up first before clients try to connect
    time.sleep(2)

    for _ in range(args.num_clients):
        Process(target=train_with_args, args=(args,)).start()

    Process(target=eval_with_args, args=(args,)).start()
