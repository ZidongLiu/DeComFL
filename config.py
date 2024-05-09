import argparse

DEFAULTS = {
    "train_batch_size": 256,
    "test_batch_size": 1000,
    "lr": 1e-4,
    "epoch": 500,
    "mu": 1e-4,
    "compressor": "quant",
    "num_pert": 1,
    "dataset": "mnist",
    "momentum": 0.9,
    "warmup_epochs": 5,
    "sparsity_file": None,
    "mask_shuffle_interval": 5,
    "grad_estimate_method": "rge-central",
    "seed": 365,
    "num_workers": 2,
    "log_to_tensorboard": None,
    "no_cuda": False,
    "no_mps": False,
    "checkpoint": None,
    "create_many_checkpoint": True,
    "checkpoint_update_plan": "every10",
    # Cezo_fl
    "iterations": 100,
    "eval_iterations": 20,
    "num_clients": 5,
    "num_sample_clients": 3,
    "local_update_steps": 1,
}


# Parameters
def get_params():
    parser = argparse.ArgumentParser(description="PyTorch training")

    # cezo-fl
    parser.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    parser.add_argument("--eval-iterations", type=int, default=DEFAULTS["eval_iterations"])
    parser.add_argument("--num-clients", type=int, default=DEFAULTS["num_clients"])
    parser.add_argument("--num-sample-clients", type=int, default=DEFAULTS["num_sample_clients"])
    parser.add_argument("--local-update-steps", type=int, default=DEFAULTS["local_update_steps"])
    # rge_main
    parser.add_argument("--train-batch-size", type=int, default=DEFAULTS["train_batch_size"])
    parser.add_argument("--test-batch-size", type=int, default=DEFAULTS["test_batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"], help="Learning rate")
    parser.add_argument("--epoch", type=int, default=DEFAULTS["epoch"])

    parser.add_argument("--mu", type=float, default=DEFAULTS["mu"])
    parser.add_argument("--compressor", type=str, default=DEFAULTS["compressor"])
    parser.add_argument("--num-pert", type=int, default=DEFAULTS["num_pert"])
    parser.add_argument("--dataset", type=str, default=DEFAULTS["dataset"])
    parser.add_argument("--momentum", type=float, default=DEFAULTS["momentum"])
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULTS["warmup_epochs"])

    parser.add_argument("--sparsity-file", type=str, default=DEFAULTS["sparsity_file"])
    parser.add_argument(
        "--mask-shuffle-interval",
        type=int,
        default=DEFAULTS["mask_shuffle_interval"],
    )

    # Rarely change
    parser.add_argument(
        "--grad-estimate-method",
        type=str,
        default=DEFAULTS["grad_estimate_method"],
        choices=["rge-central", "rge-forward", "cge-forward"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="random seed")
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument(
        "--log-to-tensorboard",
        type=str,
        default=DEFAULTS["log_to_tensorboard"],
    )

    # checkpoints
    parser.add_argument("--checkpoint", type=str, default=DEFAULTS["checkpoint"])
    parser.add_argument(
        "--create-many-checkpoint",
        default=DEFAULTS["create_many_checkpoint"],
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--checkpoint-update-plan",
        type=str,
        default=DEFAULTS["checkpoint_update_plan"],
        choices=["never", "every5", "every10", "best_loss", "best_acc"],
    )

    # No need to change
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=DEFAULTS["no_cuda"],
        help="disables CUDA training",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=DEFAULTS["no_mps"],
        help="disables macOS GPU training",
    )
    return parser


def get_args_dict(args):
    return {key: getattr(args, key) for key in DEFAULTS.keys()}


def get_args_str(args):
    # important ones, add to string regardless of it's different from default
    base_str = (
        f"{args.dataset}-lr-{args.lr}-mmtm-{args.momentum}"
        + f"-npert-{args.num_pert}-{args.grad_estimate_method}"
    )
    # only add to string if it's different from default
    advanced_items = []
    for key in ["mu", "seed", "sparsity_file", "mask_shuffle_interval"]:
        if getattr(args, key) != DEFAULTS[key]:
            v = getattr(args, key)
            if key == "sparsity_file":
                v = v.replace("/", ".").replace("\\", ".")

            advanced_items += [f"{key}-{v}"]

    if len(advanced_items):
        advanced_str = "-".join(advanced_items)
        return base_str + "-" + advanced_str

    return base_str
