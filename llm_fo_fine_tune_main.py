import torch
from tqdm import tqdm


from cezo_fl.util.metrics import Metric
from cezo_fl.fl_helpers import get_server_name
from experiment_helper import prepare_settings
from experiment_helper.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    NormalTrainingLoopSetting,
)
from experiment_helper.data import get_dataloaders
from experiment_helper.device import use_device


class CliSetting(
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    NormalTrainingLoopSetting,
):
    """
    This is a replacement for regular argparse module.
    We used a third party library pydantic_setting to make command line interface easier to manage.
    Example:
    if __name__ == "__main__":
        args = CliSetting()

    args will have all parameters defined by all components.
    """

    pass


if __name__ == "__main__":
    args = CliSetting()

    device_map = use_device(args.device_setting, 1)
    train_loaders, test_loader = get_dataloaders(
        args.data_setting, 1, args.seed, hf_model_name=args.get_hf_model_name()
    )
    device = device_map[get_server_name()]

    def inf_loader(dl):
        while True:
            for v in dl:
                yield v

    inf_test_loader = inf_loader(test_loader)

    (_, metrics) = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )

    model = prepare_settings.get_model(args.dataset, args.model_setting, args.seed)
    model.to(device)
    optimizer = prepare_settings.get_optimizer(model, args.dataset, args.optimizer_setting)

    acc = Metric("accuracy")
    model.eval()
    with torch.no_grad():
        for batch_input_dict, batch_output_tensor in test_loader:
            batch_input_dict = batch_input_dict.to("cuda")
            batch_output_tensor = batch_output_tensor.to("cuda")

            # Forward pass to get logits
            outputs = model(
                input_ids=batch_input_dict.input_ids, attention_mask=batch_input_dict.attention_mask
            )

            batch_acc = metrics.test_acc(outputs, batch_output_tensor)
            acc.update(batch_acc)
            del batch_input_dict, batch_output_tensor, outputs, batch_acc
            torch.cuda.empty_cache()
    print(f"Start, Accuracy: {acc.avg:.4f}")

    num_epochs = 20
    train_loader = train_loaders[0]
    model.train()
    total_loss = 0.0
    inf_train_loader = inf_loader(train_loader)
    eval_iterations = 200
    train_losses = []
    eval_accs = []
    for i in tqdm(range(10000)):
        batch_input_dict, batch_output_tensor = next(inf_train_loader)
        batch_input_dict = batch_input_dict.to("cuda")
        batch_output_tensor = batch_output_tensor.to("cuda")
        optimizer.zero_grad()

        # Forward pass to get logits
        outputs = model(
            input_ids=batch_input_dict.input_ids, attention_mask=batch_input_dict.attention_mask
        )

        # Calculate the loss
        loss = metrics.train_loss(outputs, batch_output_tensor)
        total_loss += loss.item()
        if (i + 1) % 50 == 0:
            print(f"Iteration: {i}, Loss: {(total_loss/50):.6f}")
            train_losses += [(i, total_loss / 50)]
            total_loss = 0.0

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Print average loss for the epoch
        average_loss = total_loss / len(train_loader)

        if (i + 1) % eval_iterations == 0:
            acc = Metric("accuracy")
            model.eval()
            with torch.no_grad():
                for batch_input_dict, batch_output_tensor in test_loader:
                    batch_input_dict = batch_input_dict.to("cuda")
                    batch_output_tensor = batch_output_tensor.to("cuda")

                    # Forward pass to get logits
                    outputs = model(
                        input_ids=batch_input_dict.input_ids,
                        attention_mask=batch_input_dict.attention_mask,
                    )

                    batch_acc = metrics.test_acc(outputs, batch_output_tensor)
                    acc.update(batch_acc)
                    del batch_input_dict, batch_output_tensor, outputs, batch_acc
                    torch.cuda.empty_cache()

            print(f"Iteration: {i}, Accuracy: {acc.avg:.4f}")
            eval_accs += [(i, acc.avg)]
