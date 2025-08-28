from huggingface_hub.repository import atexit
import torch

from cezo_grpc import sample_pb2
from cezo_grpc import data_helper

import grpc_client

from cezo_fl.util.metrics import Metric

from cezo_grpc import cli_interface

from experiment_helper import prepare_settings
from experiment_helper.device import use_device
from experiment_helper.data import get_dataloaders


def setup_eval_model(args: cli_interface.CliSetting):
    device_map = use_device(args.device_setting, args.num_clients)
    device_name = "server"
    device = device_map[device_name]

    _, test_loader = get_dataloaders(
        args.data_setting, args.num_clients, args.seed, args.get_hf_model_name()
    )
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )
    server_model_inference = model_inferences.test_inference
    server_criterion = metrics.test_loss
    server_accuracy_func = metrics.test_acc

    model = prepare_settings.get_model(
        dataset=args.dataset, model_setting=args.model_setting, seed=args.seed
    ).to(device)
    optimizer = prepare_settings.get_optimizer(
        model=model, dataset=args.dataset, optimizer_setting=args.optimizer_setting
    )
    grad_estimator = prepare_settings.get_gradient_estimator(
        model=model,
        device=device,
        rge_setting=args.rge_setting,
        model_setting=args.model_setting,
    )

    def update_model(seeds_list, grad_scalar_list):
        model.train()
        for iteration_seeds, iteration_grad_sclar in zip(seeds_list, grad_scalar_list):
            grad_estimator.update_model_given_seed_and_grad(
                optimizer,
                iteration_seeds,
                iteration_grad_sclar,
            )

    def eval_model():
        model.eval()
        eval_loss = Metric("Eval loss")
        eval_accuracy = Metric("Eval accuracy")
        with torch.no_grad():
            for _, (batch_inputs, batch_labels) in enumerate(test_loader):
                if device != torch.device("cpu") or grad_estimator.torch_dtype != torch.float32:
                    batch_inputs = batch_inputs.to(device, grad_estimator.torch_dtype)
                    # NOTE: label does not convert to dtype
                    if isinstance(batch_labels, torch.Tensor):
                        batch_labels = batch_labels.to(device)
                pred = server_model_inference(model, batch_inputs)
                eval_loss.update(server_criterion(pred, batch_labels))
                eval_accuracy.update(server_accuracy_func(pred, batch_labels))
        print(
            f"Eval Loss:{eval_loss.avg:.4f}, " f"Accuracy:{eval_accuracy.avg * 100:.2f}%",
        )
        return eval_loss.avg, eval_accuracy.avg

    return update_model, eval_model, device


def eval_with_args(args):
    ps_stub = grpc_client.get_stub()

    grpc_client.repeat_every(
        lambda: ps_stub.ConnectEval(sample_pb2.EmptyRequest()), lambda x: x.successful
    )

    print("Connected as Evaluation Client")
    # when program exits, we need to disconnect this client from server
    atexit.register(lambda: ps_stub.DisconnectEval(sample_pb2.EmptyRequest()))

    with torch.no_grad():
        update_model, eval_model, device = setup_eval_model(args)

        def try_to_eval():
            try_to_eval_result = grpc_client.repeat_every(
                lambda: ps_stub.TryToEval(sample_pb2.EmptyRequest()),
                lambda x: x.successful,
            )

            print("Start update eval model")
            pull_seeds_list = data_helper.protobuf_to_py_list_of_list_of_ints(
                try_to_eval_result.pullSeeds
            )
            raw_grad_list = data_helper.protobuf_to_py_list_of_list_of_list_of_floats(
                try_to_eval_result.pullGrads
            )
            tensor_grad_list = [
                [torch.tensor(v, device=device) for v in vv] for vv in raw_grad_list
            ]

            update_model(pull_seeds_list, tensor_grad_list)
            eval_loss, eval_accuracy = eval_model()

            print("Submit Eval Result")
            ps_stub.SubmitEvaluation(
                sample_pb2.SubmitEvaluationRequest(evalAccuracy=eval_accuracy, evalLoss=eval_loss)
            )

        grpc_client.repeat_every(try_to_eval, lambda x: False)


if __name__ == "__main__":
    args = cli_interface.CliSetting()
    print(args)
    eval_with_args(args)
