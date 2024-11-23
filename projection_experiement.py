import torch
from tqdm import tqdm

import decomfl_main
from cezo_fl.util.metrics import Metric
from config import get_params
from preprocess import preprocess
import random
import utils

args = get_params().parse_args()
args.large_model = "opt-125m"
args.dataset = "mnist"
args.train_batch_size = 32
args.test_batch_size = 32
args.num_pert = 5
args.lr = 1e-3
args.mu = 1e-3
args.grad_estimate_method = "rge-forward"


device_map, train_loaders, test_loader = preprocess(args)
device = device_map["server"]


def inf_loader(dl):
    while True:
        for v in dl:
            yield v


inf_test_loader = inf_loader(test_loader)

# large_model = args.large_model
# model_name = SUPPORTED_LLM[large_model]
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

model, criterion, optimizer, grad_estimator, accuracy_func = (
    decomfl_main.prepare_settings_underseed(args, device)
)
model.to(device)

def model_forward(batch_inputs):
    if args.dataset == 'sst2':
        outputs = model(
            input_ids=batch_input_dict.input_ids, attention_mask=batch_input_dict.attention_mask
        )
    else:
        outputs = model(batch_input_dict)
    
    return outputs


# acc = Metric("accuracy")
# model.eval()
# with torch.no_grad():
#     for batch_input_dict, batch_output_tensor in test_loader:
#         batch_input_dict = batch_input_dict.to("cuda")
#         batch_output_tensor = batch_output_tensor.to("cuda")

#         # Forward pass to get logits
#         outputs = model_forward(batch_input_dict)

#         batch_acc = accuracy_func(outputs, batch_output_tensor)
#         acc.update(batch_acc)
#         del batch_input_dict, batch_output_tensor, outputs, batch_acc
#         torch.cuda.empty_cache()
# print(f"Start, Accuracy: {acc.avg:.4f}")

num_epochs = 20
train_loader = train_loaders[0]
model.train()
total_loss = 0.0
inf_train_loader = inf_loader(train_loader)
eval_iterations = 100
train_numbers = []
loss_and_acc = []

torch.manual_seed(args.seed)
num_pert = args.num_pert


for i in tqdm(range(1000)):
    model.train()
    batch_input_dict, batch_output_tensor = next(inf_train_loader)
    batch_input_dict = batch_input_dict.to(device)
    batch_output_tensor = batch_output_tensor.to(device)

    # STEP 1: Generate perturbations
    rng_seed = random.randint(0, 1000000)
    rng = torch.Generator(device="cuda").manual_seed(rng_seed)
    perturbations = grad_estimator.generate_mutiple_perturbation_norm(rng)

    # STEP 2: Calculate the loss and accuracy
    

    # STEP 3: Get true grad Y = loss.backward()
    # ----
    optimizer.zero_grad()
    outputs = model_forward(batch_input_dict)
    loss = criterion(outputs, batch_output_tensor)
    loss.backward()
    true_grad = utils.get_flatten_model_grad(model)
    true_grad_l2 = true_grad.norm(p=2)
    # ----
    # STEP 4: Calculate XtY, this is simple projection facor. Estimator=XXtY
    # ----
    with torch.no_grad():
        simple_projection_factor = [true_grad.inner(p) for p in perturbations]
        simple_projection_grad = sum(
            [simple_projection_factor[i] * perturbations[i] for i in range(num_pert)]
        )
        simple_projection_l2 = simple_projection_grad.norm(p=2)
        simple_projection_scaling_factor = true_grad_l2 / simple_projection_l2
        scaled_simple_projection_grad = simple_projection_scaling_factor * simple_projection_grad
        simple_projection_diff = (true_grad - scaled_simple_projection_grad).norm(p=2)
    # ----
    # X = torch.stack(perturbations, dim = 1)
    # simple_projection_factor = torch.matmul(X, true_grad.reshape(-1,1))
    # simple_projection_grad = torch.matmul(X, simple_projection_factor)
    # simple_projection_grad_l2 = simple_projection_grad.norm(dim=1, p=2)

    # STEP 5: Calculate XtX-1 × XtY, this is true projection factor. Estimator=X(XtX)-1Y
    # inner_X = torch.matmul(X.t(), X)
    # true_projection_factor = torch.matmul(inner_X.inverse(), simple_projection_factor)
    # true_projection_factor_grad = torch.matmul(X, true_projection_factor)
    # true_projection_factor_grad_l2 = true_projection_factor_grad.norm(dim=1, p=2)

    # STEP 6: Calculate zo factor using rge. Estimator=X×Vzo
    # ----
    optimizer.zero_grad()
    with torch.no_grad():
        ZO_factor = grad_estimator.compute_grad(
            batch_input_dict, batch_output_tensor, criterion, rng_seed
        )
        ZO_grad = sum(perturbations[i] * ZO_factor[i] for i in range(num_pert))
        ZO_grad_l2 = ZO_grad.norm(p=2)
        ZO_grad_scaling_factor = true_grad_l2 / ZO_grad_l2
        scaled_ZO_grad = ZO_grad_scaling_factor * ZO_grad
        ZO_grad_diff = (true_grad - scaled_ZO_grad).norm(p=2)

    train_numbers.append(
        {
            "iteration": i,
            "simple_projection_scaling_factor": simple_projection_scaling_factor.item(),
            "simple_projection_diff": simple_projection_diff.item(),
            "ZO_grad_scaling_factor": ZO_grad_scaling_factor.item(),
            "ZO_grad_diff": ZO_grad_diff.item(),
            'true_grad_l2': true_grad_l2.item()
        }
    )
    # ----
    # Backward pass and optimization step
    optimizer.zero_grad()
    outputs = model_forward(batch_input_dict)
    loss = criterion(outputs, batch_output_tensor)
    loss.backward()
    optimizer.step()

    # Print average loss for the epoch
    total_loss += loss.item()
    if (i + 1) % eval_iterations == 0:
        acc = Metric("accuracy")
        model.eval()
        with torch.no_grad():
            for batch_input_dict, batch_output_tensor in test_loader:
                batch_input_dict = batch_input_dict.to("cuda")
                batch_output_tensor = batch_output_tensor.to("cuda")

                # Forward pass to get logits
                outputs = model_forward(batch_input_dict)

                batch_acc = accuracy_func(outputs, batch_output_tensor)
                acc.update(batch_acc)
                del batch_input_dict, batch_output_tensor, outputs, batch_acc
                torch.cuda.empty_cache()
        log_loss = total_loss / eval_iterations
        print(f"Iteration: {i}, Loss: {(log_loss):.6f}, Accuracy: {acc.avg:.4f}")
        loss_and_acc.append({"iteration": i, "loss": log_loss, "acc": acc.avg})
        total_loss = 0.0


import pandas as pd
from matplotlib import pyplot as plt


numbers_df = pd.DataFrame(train_numbers)
plt.figure()
plt.title(f'{args.dataset} l2 diff')
plt.hist(numbers_df['ZO_grad_diff'], bins=30, alpha = 0.5, label='ZO')
plt.hist(numbers_df['simple_projection_diff'], bins=30, alpha = 0.5, label='simple projection')
plt.hist(numbers_df['true_grad_l2'], bins=30, alpha = 0.5, label='true_grad_l2')
plt.legend()

plt.figure()
plt.title(f'{args.dataset} scaling factor')
plt.hist(numbers_df['ZO_grad_scaling_factor'], bins=30, alpha = 0.5, label='ZO')
plt.hist(numbers_df['simple_projection_scaling_factor'], bins=30, alpha = 0.5, label='simple projection')

plt.legend()

plt.figure()
plt.title(f'{args.dataset} scaling factor logX')
plt.hist(numbers_df['ZO_grad_scaling_factor'], bins=30, alpha = 0.5, label='ZO')
plt.hist(numbers_df['simple_projection_scaling_factor'], bins=30, alpha = 0.5, label='simple projection')
plt.xscale('log')
plt.legend()
