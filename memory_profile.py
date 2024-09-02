import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
from os import path
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_params, get_args_str
from preprocess import preprocess_cezo_fl

from cezo_fl.server import CeZO_Server
from cezo_fl.client import Client

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



MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000
torch.cuda.memory._record_memory_history(
    max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
)

args = get_params().parse_args()


args.dataset = "sst2"
args.lr = 1e-7
args.momentum = 0
args.seed = 365
args.num_clients = 1
args.train_batch_size = 8
args.test_batch_size = 10

device, train_loaders, test_loader = preprocess_cezo_fl(args)


def inf_loader(dl):
    while True:
        for v in dl:
            yield v


inf_test_loader = inf_loader(test_loader)


model, criterion, optimizer, grad_estimator, accuracy_func = prepare_settings_underseed(
    args, device
)
model_precision = 'half'
if model_precision == 'half':
    model.half()
model.to(device)

n_round = 5
pred = False
clean_each_round = False
acc = Metric("accuracy")

model.eval()
with torch.no_grad():
    for i in range(n_round):
        batch_input_dict, batch_output_tensor = next(inf_test_loader)
        batch_input_dict = batch_input_dict.to("cuda")
        batch_output_tensor = batch_output_tensor.to("cuda")

        # Forward pass to get logits
        if pred:
            outputs = model(
                input_ids=batch_input_dict.input_ids, attention_mask=batch_input_dict.attention_mask
            )

            batch_acc = accuracy_func(outputs, batch_output_tensor)
            acc.update(batch_acc)
            if clean_each_round:
                del batch_input_dict, batch_output_tensor, outputs, batch_acc
                torch.cuda.empty_cache()

file_prefix = f'{args.dataset}-{model_precision}-pred-{pred}-clean_each_round-{clean_each_round}'
snapshot_name = f'./memory_snapshots/{file_prefix}.pickle'
torch.cuda.memory._dump_snapshot(snapshot_name)


# Stop recording memory snapshot history.
torch.cuda.memory._record_memory_history(enabled=None)