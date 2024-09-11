import torch
from tqdm import tqdm

from config import get_params
from preprocess import preprocess

from shared.metrics import Metric
import decomfl_main

args = get_params().parse_args()

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
