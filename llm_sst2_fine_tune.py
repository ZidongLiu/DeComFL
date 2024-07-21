import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
from os import path
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_params, get_args_str
from preprocess import preprocess

from cezo_fl.server import CeZO_Server
from cezo_fl.client import ResetClient

from shared.model_helpers import get_current_datetime_str
from models.cnn_mnist import CNN_MNIST
from models.lenet import LeNet
from models.cnn_fashion import CNN_FMNIST
from models.lstm import CharLSTM
from shared.language_utils import LM_TEMPLATE_MAP

from tqdm import tqdm
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
import datasets
from shared.metrics import Metric

from cezo_fl_main import prepare_settings_underseed

args = get_params().parse_args()


args.dataset = "cb"
args.lr = 1e-7
args.momentum = 0
args.seed = 365
args.num_clients = 1
args.train_batch_size = 8
args.test_batch_size = 10

device_map, train_loaders, test_loader = preprocess(args)
device = device_map["server"]


def inf_loader(dl):
    while True:
        for v in dl:
            yield v


inf_test_loader = inf_loader(test_loader)


# model_name = "facebook/opt-1.3b"
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name, padding_side="left", truncate_side="left"
# )

# template = RTETemplate()
# template = LM_TEMPLATE_MAP[args.dataset]()
# test_sample = next(inf_test_loader)
# print(test_sample[1][:, -1])
# template.get_verbalizer_id(tokenizer)
# server = setup_server_and_clients(args, device, train_loaders)

# args_str = get_args_str(args) + "-" + server.server_model.model_name

model, criterion, optimizer, grad_estimator, accuracy_func = prepare_settings_underseed(
    args, device
)
model.to(device)

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

        batch_acc = accuracy_func(outputs, batch_output_tensor)
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
    loss = criterion(outputs, batch_output_tensor)
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

                batch_acc = accuracy_func(outputs, batch_output_tensor)
                acc.update(batch_acc)
                del batch_input_dict, batch_output_tensor, outputs, batch_acc
                torch.cuda.empty_cache()

        print(f"Iteration: {i}, Accuracy: {acc.avg:.4f}")
        eval_accs += [(i, acc.avg)]
