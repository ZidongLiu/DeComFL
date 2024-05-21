import torch
from enum import Enum
from functools import partial
from dataclasses import dataclass

# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    """returns one-hot vector with given size and value 1 at given index"""
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    """returns one-hot representation of given letter"""
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    """returns a list of character indices

    Args:
        word: string

    Return:
        indices: int list with length len(word)
    """
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


class CustomLMDataset(torch.utils.data.DataLoader):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_text = self.texts[idx]
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        # left_truncation
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
        return torch.tensor(input_ids, dtype=torch.long)


class SST2Template:
    verbalizer = {0: " bad", 1: " good"}

    def verbalize_for_pred(self, sample):
        text = sample["sentence"].strip()
        return f"{text} It was"

    def verbalize(self, sample):
        label = sample["label"]
        return f"{self.verbalize_for_pred(sample)}{self.verbalizer[label]}"

    def get_verbalizer_id(self, tokenizer):
        return {k: tokenizer.encode(v)[-1] for k, v in self.verbalizer.items()}


class RTETemplate:
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No"}

    def verbalize_for_pred(self, sample):
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        return f'{premise}\nDoes this mean that "{hypothesis}" is true? Yes or No?\n'

    def verbalize(self, sample):
        label = sample["label"]
        return f"{self.verbalize_for_pred(sample)}{self.verbalizer[label]}"

    def get_verbalizer_id(self, tokenizer):
        return {k: tokenizer.encode(v)[-1] for k, v in self.verbalizer.items()}


class LmTask(Enum):
    sst2 = "sst2"
    rte = "rte"


LM_TEMPLATE_MAP = {LmTask.sst2.name: SST2Template, LmTask.rte.name: RTETemplate}


@dataclass
class LLMBatchInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


def get_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        # Pad sequences to the max length in the batch
        padded_batch = tokenizer.pad(
            {"input_ids": batch}, padding=True, max_length=max_length, return_tensors="pt"
        )
        input_ids = padded_batch["input_ids"]
        attention_mask = padded_batch["attention_mask"]
        return (
            LLMBatchInput(input_ids[:, :(-1)], attention_mask[:, :(-1)]),
            input_ids[:, 1:],
        )  # Prepare input and target sequences

    return collate_fn


class LossType(Enum):
    full_sentence = "full_sentence"
    last_token = "last_token"
    accuracy = "accuracy"


def get_lm_loss(loss_type: LossType, verbalizer_id_map: dict[int, int]):
    n_candidate = len(verbalizer_id_map)
    verbalizer_id_list = [verbalizer_id_map[i] for i in range(n_candidate)]

    if loss_type == "full_sentence":
        return full_sentence_cross_entropy_loss
    elif loss_type == "last_token":
        return partial(
            last_token_cross_entropy_loss,
            verbalizer_id_map=verbalizer_id_map,
            verbalizer_id_list=verbalizer_id_list,
        )
    elif loss_type == "accuracy":
        return partial(
            last_token_accuracy,
            verbalizer_id_map=verbalizer_id_map,
            verbalizer_id_list=verbalizer_id_list,
        )


def full_sentence_cross_entropy_loss(batch_pred, sentence_label_tokens):
    logits = batch_pred.logits
    # Flatten the logits and labels for calculating loss
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = sentence_label_tokens.contiguous().view(-1)

    # Calculate the loss
    loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat)
    return loss


def last_token_cross_entropy_loss(
    batch_pred, sentence_label_tokens, verbalizer_id_map, verbalizer_id_list
):
    logits = batch_pred.logits
    last_token_batch_pred = logits[:, -1, verbalizer_id_list].view(-1, len(verbalizer_id_list))
    last_token_label = (sentence_label_tokens[:, -1] == verbalizer_id_map[1]).to(int)

    loss = torch.nn.functional.cross_entropy(last_token_batch_pred, last_token_label)
    return loss


def last_token_accuracy(batch_pred, sentence_label_tokens, verbalizer_id_map, verbalizer_id_list):
    logits = batch_pred.logits
    last_token_batch_pred = logits[:, -1, verbalizer_id_list].view(-1, len(verbalizer_id_list))
    last_token_label = (sentence_label_tokens[:, -1] == verbalizer_id_map[1]).to(int)

    pred = last_token_batch_pred.max(1, keepdim=True)[1]
    return pred.eq(last_token_label.view_as(pred)).cpu().float().mean()
