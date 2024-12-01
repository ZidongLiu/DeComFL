import functools
from os import path
import torch


from tensorboardX import SummaryWriter
from tqdm import tqdm

from byzantine import aggregation as byz_agg
from byzantine import attack as byz_attack
from cezo_fl.client import ResetClient
from cezo_fl.fl_helpers import get_client_name
from cezo_fl.server import CeZO_Server
from cezo_fl.util import model_helpers, prepare_settings

from config import get_args_str, get_params
from preprocess import preprocess


def setup_server_and_clients(
    args, device_map: dict[str, torch.device], train_loaders
) -> CeZO_Server:
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, prepare_settings.SUPPORTED_LLM.get(args.large_model)
    )
    clients = []

    for i in range(args.num_clients):
        client_name = get_client_name(i)
        client_device = device_map[client_name]
        (
            client_model,
            client_optimizer,
            client_grad_estimator,
        ) = prepare_settings.prepare_settings_underseed(args, client_device)
        client_model.to(client_device)

        client = ResetClient(
            client_model,
            model_inferences.train_inference,
            train_loaders[i],
            client_grad_estimator,
            client_optimizer,
            metrics.train_loss,
            metrics.train_acc,
            client_device,
        )
        clients.append(client)

    server_device = device_map["server"]
    server = CeZO_Server(
        clients,
        server_device,
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_update_steps,
    )

    # set server tools
    (
        server_model,
        server_optimizer,
        server_grad_estimator,
    ) = prepare_settings.prepare_settings_underseed(args, server_device)

    server_model.to(server_device)
    server.set_server_model_and_criterion(
        server_model,
        model_inferences.test_inference,
        metrics.test_loss,
        metrics.test_acc,
        server_optimizer,
        server_grad_estimator,
    )

    # TODO(lizhe) move this into a seperate main file.
    # Prepare the Byzantine attack
    if args.byz_type == "no_byz":
        server.register_attack_func(byz_attack.no_byz)
    elif args.byz_type == "gaussian":
        server.register_attack_func(
            functools.partial(byz_attack.gaussian_attack, num_attack=args.num_byz)
        )
    elif args.byz_type == "sign":
        server.register_attack_func(
            functools.partial(byz_attack.sign_attack, num_attack=args.num_byz)
        )
    elif args.byz_type == "trim":
        server.register_attack_func(
            functools.partial(byz_attack.trim_attack, num_attack=args.num_byz)
        )
    elif args.byz_type == "krum":
        server.register_attack_func(
            functools.partial(byz_attack.krum_attack, f=args.num_byz, lr=args.lr)
        )
    else:
        raise Exception(
            "byz_type should be one of no_byz, gaussian, sign, trim, krum."
            + f"But get {args.byz_type}"
        )

    if args.aggregation == "mean":
        server.register_aggregation_func(byz_agg.mean)
    elif args.aggregation == "median":
        server.register_aggregation_func(byz_agg.median)
    elif args.aggregation == "trim":
        server.register_aggregation_func(byz_agg.trim)
    elif args.aggregation == "krum":
        server.register_aggregation_func(byz_agg.krum)
    else:
        raise Exception(
            "aggregation type should be one of mean, median, trim, krum. "
            + f"But get {args.aggregation}"
        )

    return server


# get_warmup_lr is not used for now.
def get_warmup_lr(args, current_epoch: int, current_iter: int, iters_per_epoch: int) -> float:
    assert isinstance(args.lr, float) and isinstance(args.warmup_epochs, int)
    overall_iterations = args.warmup_epochs * iters_per_epoch + 1
    current_iterations = current_epoch * iters_per_epoch + current_iter + 1
    return args.lr * current_iterations / overall_iterations


def get_size_of_model(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


if __name__ == "__main__":
    args = get_params().parse_args()
    if args.dataset == "shakespeare":
        args.num_clients = 139
    print(args)
    device_map, train_loaders, test_loader = preprocess(args)

    server = setup_server_and_clients(args, device_map, train_loaders)

    if args.log_to_tensorboard:
        assert server.server_model
        tensorboard_sub_folder = "-".join(
            [
                get_args_str(args),
                server.server_model.model_name,
                model_helpers.get_current_datetime_str(),
            ]
        )
        writer = SummaryWriter(
            path.join(
                "tensorboards",
                "decomfl",
                args.dataset,
                args.log_to_tensorboard,
                tensorboard_sub_folder,
            )
        )

    with tqdm(total=args.iterations, desc="Training:") as t, torch.no_grad():
        for ite in range(args.iterations):
            step_loss, step_accuracy = server.train_one_step(ite)
            t.set_postfix({"Loss": step_loss, "Accuracy": step_accuracy})
            t.update(1)
            if args.adjust_perturb:
                if ite == 500:
                    server.set_learning_rate(args.lr * 0.8)
                    server.set_perturbation(args.num_pert * 2)
                elif ite == 1000:
                    server.set_learning_rate(args.lr * 0.5)
                    server.set_perturbation(args.num_pert * 4)
                elif ite == 2000:
                    server.set_learning_rate(args.lr * 0.3)
                    server.set_perturbation(args.num_pert * 8)

            if args.log_to_tensorboard:
                writer.add_scalar("Loss/train", step_loss, ite)
                writer.add_scalar("Accuracy/train", step_accuracy, ite)
            # eval
            if args.eval_iterations != 0 and (ite + 1) % args.eval_iterations == 0:
                eval_loss, eval_accuracy = server.eval_model(test_loader)
                if args.log_to_tensorboard:
                    writer.add_scalar("Loss/test", eval_loss, ite)
                    writer.add_scalar("Accuracy/test", eval_accuracy, ite)
