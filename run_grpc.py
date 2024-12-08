from multiprocessing import Process

from grpc_client import train_with_args
from grpc_eval_client import eval_with_args
from grpc_server import serve
import config

import time

if __name__ == "__main__":
    args = config.get_params_grpc().parse_args()
    if args.dataset == "shakespeare":
        args.num_clients = 139

    Process(target=serve, args=(args,)).start()

    # HACK: sleep 2 seconds to make sure server can spin up first before clients try to connect
    time.sleep(2)

    for _ in range(args.num_clients):
        Process(target=train_with_args, args=(args,)).start()

    Process(target=eval_with_args, args=(args,)).start()
