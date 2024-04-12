import argparse


# Parameters
def get_params(dataset: str):
    parser = argparse.ArgumentParser(description=f"PyTorch training")
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--grad-estimate-method", type=str, default="central")
    parser.add_argument("--mu", type=float, default=1e-4)
    parser.add_argument("--compressor", type=str, default="quant")
    parser.add_argument("--num-pert", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--momentum", type=float, default=0.1)

    # Rarely change
    parser.add_argument("--seed", type=int, default=365, help="random seed")
    parser.add_argument("--num-workers", type=int, default=2)

    # No need to change
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    return parser
