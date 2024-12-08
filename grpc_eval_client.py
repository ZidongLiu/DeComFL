from huggingface_hub.repository import atexit
import torch

from cezo_grpc import sample_pb2
from cezo_grpc import data_helper

import grpc_client

import config
import preprocess
import decomfl_main

from cezo_fl.util.metrics import Metric


def setup_eval_model(args):
    device_map, _, test_loader = preprocess.preprocess(args)
    device = device_map["server"]
    (
        model,
        criterion,
        optimizer,
        grad_estimator,
        accuracy_func,
    ) = decomfl_main.prepare_settings_underseed(args, device)

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
                    batch_labels = batch_labels.to(device)
                pred = grad_estimator.model_forward(batch_inputs)
                eval_loss.update(criterion(pred, batch_labels))
                eval_accuracy.update(accuracy_func(pred, batch_labels))
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
    args = config.get_params_grpc().parse_args()
    if args.dataset == "shakespeare":
        args.num_clients = 139
    print(args)

    eval_with_args(args)
