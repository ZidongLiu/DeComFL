from os import path
from typing import Any
import torch
import torch.nn as nn
from cezo_fl.coordinate_gradient_estimator import CoordinateGradientEstimator as CGE
from cezo_fl.models.cnn_fashion import CNN_FMNIST
from cezo_fl.models.cnn_mnist import CNN_MNIST
from cezo_fl.models.lenet import LeNet
from cezo_fl.models.lstm import CharLSTM
from cezo_fl.util.checkpoint import CheckPoint
from cezo_fl.util.metrics import Metric, accuracy
from config import get_args_str, get_params
from preprocess import preprocess, use_sparsity_dict

from peft import LoraConfig, get_peft_model
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from cezo_fl.random_gradient_estimator import RandomGradientEstimator as RGE
from cezo_fl.util import model_helpers
from cezo_fl.util.language_utils import LM_TEMPLATE_MAP, SUPPORTED_LLM, get_lm_loss


def prepare_settings(args, device):
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.model_dtype]
    torch.manual_seed(args.seed)
    if args.dataset == "mnist":
        model = CNN_MNIST().to(torch_dtype).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            weight_decay=1e-5,
            momentum=args.momentum,
        )
        accuracy_func = accuracy
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    elif args.dataset == "cifar10":
        model = LeNet().to(torch_dtype).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            weight_decay=5e-4,
            momentum=args.momentum,
        )
        accuracy_func = accuracy
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[200], gamma=0.1
        # )
    elif args.dataset == "fashion":
        model = CNN_FMNIST().to(torch_dtype).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            weight_decay=1e-5,
            momentum=args.momentum,
        )
        accuracy_func = accuracy
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[200], gamma=0.1
        # )
    elif args.dataset == "shakespeare":
        model = CharLSTM().to(torch_dtype).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        accuracy_func = accuracy
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[200], gamma=0.1
        # )
    elif args.dataset in LM_TEMPLATE_MAP.keys():
        large_model = args.large_model
        model_name = SUPPORTED_LLM[large_model]
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
        model.model_name = large_model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncate_side="left"
        )
        template = LM_TEMPLATE_MAP[args.dataset]()
        if args.dataset in ["sst2", "cb", "wsc", "wic", "multirc", "rte", "boolq", "gen"]:
            if args.lora:
                # this step initialize lora parameters, which should be under control of seed
                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                )
                model = get_peft_model(model, lora_config).to(torch_dtype)
            verbalizer_id_map = template.get_verbalizer_id(tokenizer)
            criterion = get_lm_loss("last_token", verbalizer_id_map=verbalizer_id_map)
            optimizer = torch.optim.SGD(
                model_helpers.get_trainable_model_parameters(model),
                lr=args.lr,
                momentum=0,
                weight_decay=5e-4,
            )
            accuracy_func = get_lm_loss("accuracy", verbalizer_id_map=verbalizer_id_map)
        elif args.dataset in ["squad", "drop", "xsum"]:
            criterion = get_lm_loss("full_sentence", verbalizer_id_map={})
            optimizer = torch.optim.SGD(
                model_helpers.get_trainable_model_parameters(model),
                lr=args.lr,
                momentum=0,
                weight_decay=0,
            )
            accuracy_func = get_lm_loss("f1", tokenizer=tokenizer)
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported")
    else:
        raise Exception(f"Dataset {args.dataset} is not supported")

    if args.grad_estimate_method in ["rge-central", "rge-forward"]:
        method = args.grad_estimate_method[4:]
        print(f"Using RGE {method}")
        if args.dataset in ["squad", "drop"]:
            generation_mode = True
            generation_mode_kwargs = {
                "do_sample": True,
                "temperature": 1.0,
                "num_beams": 2,
                "top_p": 0.3,
                "top_k": None,
                "num_return_sequences": 1,
                "max_new_tokens": 5,  # will be adjusted dynamically later
                "max_length": 2048,
                "length_penalty": 2,
                "early_stopping": True,
                "eos_token_id": [
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                    tokenizer.eos_token_id,
                ],
            }
        elif args.dataset in ["xsum"]:
            generation_mode = True
            generation_mode_kwargs = {
                "do_sample": True,
                "temperature": 1.0,
                "num_beams": 2,
                "top_p": 0.95,
                "top_k": None,
                "num_return_sequences": 1,
                "max_new_tokens": 500,  # will be adjusted dynamically later
                "max_length": 2048,
                "early_stopping": True,
                "eos_token_id": [
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                    tokenizer.eos_token_id,
                ],
            }
        else:
            generation_mode = False
            generation_mode_kwargs = None
        grad_estimator = RGE(
            model,
            parameters=model_helpers.get_trainable_model_parameters(model),
            mu=args.mu,
            num_pert=args.num_pert,
            grad_estimate_method=method,
            device=device,
            torch_dtype=torch_dtype,
            # To save memory consumption, we have to use parameter-wise perturb + no_optim together.
            sgd_only_no_optim=args.no_optim,
            paramwise_perturb=args.no_optim,
            # For generation mode, the forward style is different
            generation_mode=generation_mode,
            generation_mode_kwargs=generation_mode_kwargs,
        )
    else:
        raise Exception(f"Grad estimate method {args.grad_estimate_method} not supported")
    return model, criterion, optimizer, grad_estimator, accuracy_func


def inf_loader(dl):
    while True:
        for v in dl:
            yield v


def train_model(ite: int) -> tuple[float, float]:
    model.train()
    train_loss = Metric("train loss")
    train_accuracy = Metric("train accuracy")
    inf_train_loader = inf_loader(train_loader)
    images, labels = next(inf_train_loader)
    if device != torch.device("cpu"):
        images, labels = images.to(device), labels.to(device)
    grad_estimator.generation_mode = False
    # update models
    if args.no_optim:
        with torch.no_grad():
            seed = ite**2 + ite
            grad_scalars = grad_estimator._zo_grad_estimate_paramwise(
                images, labels, criterion, seed
            )
            grad_estimator.update_model_given_seed_and_grad(optimizer, [seed], [grad_scalars])
    else:
        optimizer.zero_grad()
        grad_estimator.compute_grad(images, labels, criterion, seed=ite**2 + ite)
        optimizer.step()

    pred = grad_estimator.model_forward(images)
    train_loss.update(criterion(pred, labels))
    if args.dataset not in ["squad", "drop", "xsum"]:
        train_accuracy.update(accuracy_func(pred, labels))
    else:
        train_accuracy.update(torch.tensor(0))  # No train accuracy for generation task
    return train_loss.avg, train_accuracy.avg


def eval_model(ite: int) -> tuple[float, float]:
    model.eval()
    eval_loss = Metric("Eval loss")
    eval_accuracy = Metric("Eval accuracy")
    if args.dataset in ["squad", "drop", "xsum"]:
        grad_estimator.generation_mode = True

    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader):
            if device != torch.device("cpu"):
                images = images.to(device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
            pred = grad_estimator.model_forward(images)

            if args.dataset not in ["squad", "drop", "xsum"]:
                eval_loss.update(criterion(pred, labels))
            else:
                eval_loss.update(torch.tensor(0))  # No eval loss for generation task
            eval_accuracy.update(accuracy_func(pred, labels))
    print(
        f"Evaluation(round {ite}): Eval Loss:{eval_loss.avg:.4f}, "
        f"Accuracy:{eval_accuracy.avg * 100:.2f}%"
    )
    return eval_loss.avg, eval_accuracy.avg


if __name__ == "__main__":
    args = get_params().parse_args()
    torch.manual_seed(args.seed)

    # set num_clients = 1 to make sure there's 1 train_loader
    args.num_clients = 1
    device_map, train_loaders, test_loader = preprocess(args)
    train_loader = train_loaders[0]
    device = device_map["server"]

    model, criterion, optimizer, grad_estimator, accuracy_func = prepare_settings(args, device)
    # No checkpoint to save the memory
    # checkpoint = CheckPoint(args, model, optimizer, grad_estimator)

    args_str = get_args_str(args) + "-" + model.model_name
    if args.log_to_tensorboard:
        tensorboard_sub_folder = args_str + "-" + model_helpers.get_current_datetime_str()
        writer = SummaryWriter(
            path.join(
                "tensorboards",
                args.dataset,
                args.log_to_tensorboard,
                tensorboard_sub_folder,
            )
        )
    with tqdm(total=args.iterations, desc="Training:") as t, torch.no_grad():
        sparsity_dict = use_sparsity_dict(args, model.model_name)
        for ite in range(args.iterations):
            if sparsity_dict is not None and ite % args.mask_shuffle_interval == 0:
                raise NotImplementedError("We no longer support pruning mask.")

            train_loss, train_accuracy = train_model(ite)
            if args.log_to_tensorboard:
                writer.add_scalar("Loss/train", train_loss, ite)
                writer.add_scalar("Accuracy/train", train_accuracy, ite)
            if args.eval_iterations != 0 and (ite + 1) % args.eval_iterations == 0:
                eval_loss, eval_accuracy = eval_model(ite)
                if args.log_to_tensorboard:
                    writer.add_scalar("Loss/test", eval_loss, ite)
                    writer.add_scalar("Accuracy/test", eval_accuracy, ite)
            t.set_postfix({"Loss": train_loss, "Accuracy": train_accuracy})
            t.update(1)
        if args.log_to_tensorboard:
            writer.close()

peak_memory = torch.cuda.max_memory_allocated(device="cuda:0")
print(f"Peak memory usage: {peak_memory / (1024 ** 3):.2f} GB")
