from multiprocessing import Process

from grpc_client import train_with_args
from grpc_eval_client import eval_with_args
from grpc_server import serve
import config


if __name__ == "__main__":
    args = config.get_params().parse_args()
    if args.dataset == "shakespeare":
        args.num_clients = 139

    Process(target=serve, args=(args,)).start()

    for _ in range(args.num_clients):
        Process(target=train_with_args, args=(args,)).start()

    Process(target=eval_with_args, args=(args,)).start()


# from multiprocessing import Process
# import os


# def info(title):
#     print(title)
#     print("module name:", __name__)
#     print("parent process:", os.getppid())
#     print("process id:", os.getpid())


# def f(name):
#     info("function f")
#     print("hello", name)


# if __name__ == "__main__":
#     info("main line")
#     p = Process(target=f, args=("bob",))
#     p.start()
#     p.join()
