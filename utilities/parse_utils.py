import argparse
import math
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Decision-Preserving Regularization (DPR) from scratch with cached features.")
    parser.add_argument("--outpath", "-op", default=None,
                        help="output directory; default: <repository>/logs/<dataset>_t<n_task>")
    parser.add_argument("--no_save_checkpoints", action="store_true",
                        help="skip per-task model checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nw", type=int, default=12, help="data loader workers")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--dataset", default="cifar100",
                        choices=["cifar100", "tiny-imagenet", "imagenet-subset"])
    parser.add_argument("--data_path", default="./data", help="directory containing the dataset")
    parser.add_argument("--n_task", type=int, default=10)
    parser.add_argument("--n_class_first_task", type=int, default=-1,
                        help="classes in the first task; -1 splits all classes equally")
    parser.add_argument("--valid_size", type=float, default=0.0)
    parser.add_argument("--epochs_first_task", type=int, default=None,
                        help="first-task epochs: 160 for ImageNet-Subset, 100 otherwise")
    parser.add_argument("--epochs_next_task", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18"])
    parser.add_argument("--lambda_s", type=float, default=0.5, help="decision subspace coefficient (lambda_1)")
    parser.add_argument("--lambda_c", type=float, default=0.01, help="orthogonal complement coefficient (lambda_2)")
    parser.add_argument("--transport_ridge", type=float, default=1e-3)
    parser.add_argument("--transport_views", type=int, choices=[1, 2, 4], default=None,
                        help="affine fitting views: 4 for CIFAR-100, 1 otherwise")
    parser.add_argument("--balanced_bs", type=int, default=256, help="classifier calibration batch size")
    parser.add_argument("--balanced_epochs", type=int, default=50, help="classifier calibration epochs")
    parser.add_argument("--balanced_lr", type=float, default=1e-3, help="classifier calibration learning rate")
    parser.add_argument("--max_tasks", type=int, default=-1,
                        help="stop after N tasks for a short run; -1 trains all tasks")
    args = parser.parse_args()
    default_outpath = str(Path(__file__).resolve().parents[1] / "logs" / f"{args.dataset}_t{args.n_task}")
    if args.outpath is None:
        args.outpath = default_outpath
    if args.epochs_first_task is None:
        args.epochs_first_task = 160 if args.dataset == "imagenet-subset" else 100
    if args.transport_views is None:
        args.transport_views = 4 if args.dataset == "cifar100" else 1
    if any(not math.isfinite(v) or v < 0 for v in
           (args.lambda_s, args.lambda_c)):
        parser.error("lambda_s and lambda_c must be finite and non-negative")
    if not math.isfinite(args.transport_ridge) or args.transport_ridge <= 0:
        parser.error("transport_ridge must be finite and positive")
    if args.n_task < 2:
        parser.error("n_task must be at least 2")
    return args
