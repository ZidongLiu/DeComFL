import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


from cezo_fl.util import model_helpers
from cezo_fl.models.cnn_fashion import CNN_FMNIST
from cezo_fl.models.cnn_mnist import CNN_MNIST
from cezo_fl.models.lenet import LeNet
from cezo_fl.models.lstm import CharLSTM
from cezo_fl.random_gradient_estimator import RandomGradientEstimator as RGE
from cezo_fl.util.language_utils import LM_TEMPLATE_MAP, SUPPORTED_LLM, get_lm_loss
from cezo_fl.util.metrics import accuracy


def prepare_settings_underseed(args, device, server_or_client: str = "server"):
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
        if args.dataset in ["sst2", "cb", "wsc", "wic", "multirc", "rte", "boolq"]:
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
            if server_or_client == "server":
                criterion = get_lm_loss("f1", tokenizer=tokenizer)
                optimizer = torch.optim.SGD(
                    model_helpers.get_trainable_model_parameters(model),
                    lr=args.lr,
                    momentum=0,
                    weight_decay=0,
                )
                accuracy_func = get_lm_loss("f1", tokenizer=tokenizer)
            elif server_or_client == "client":
                criterion = get_lm_loss("full_sentence", verbalizer_id_map={})
                optimizer = torch.optim.SGD(
                    model_helpers.get_trainable_model_parameters(model),
                    lr=args.lr,
                    momentum=0,
                    weight_decay=0,
                )
                accuracy_func = get_lm_loss("full_sentence", verbalizer_id_map={})
            else:
                raise ValueError(
                    "server_or_client must be either 'server' or 'client'. "
                    f"But get {server_or_client}"
                )
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported")
    else:
        raise Exception(f"Dataset {args.dataset} is not supported")

    if args.grad_estimate_method in ["rge-central", "rge-forward"]:
        method = args.grad_estimate_method[4:]
        print(f"Using RGE {method}")
        if args.dataset in ["squad", "drop"] and server_or_client == "server":
            generation_mode = True
            # TODO move this setting partially to the args
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
        elif args.dataset in ["xsum"] and server_or_client == "server":
            generation_mode = True
            # TODO move this setting partially to the args
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
