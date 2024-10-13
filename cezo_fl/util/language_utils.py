from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Literal, Sequence
from collections import Counter
import re
import string
import torch
import numpy as np

# utils for shakespeare dataset

ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)
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


# LLM
SUPPORTED_LLM = {
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-125m": "facebook/opt-125m",
}


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


class CustomLMGenerationDataset(torch.utils.data.DataLoader):
    def __init__(self, texts, golds, tokenizer, max_length):
        assert len(texts) == len(golds)
        self.texts = texts
        self.golds = golds
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
        return torch.tensor(input_ids, dtype=torch.long), (
            len(input_ids),
            self.golds[idx],
        )


class Template:
    def encode(self, sample):
        """
        Return prompted version of the example (without the answer/candidate)
        """
        raise NotImplementedError

    def verbalize(self, sample):
        """
        Return the prompted version of the example (with the answer/candidate)
        """
        raise NotImplementedError


class ClassificationTemplate(Template):
    verbalizer = {0: "0", 1: "1"}

    def get_verbalizer_id(self, tokenizer):
        return {k: tokenizer.encode(v)[-1] for k, v in self.verbalizer.items()}

    def verbalize(self, sample):
        label = sample["label"]
        return f"{self.verbalize_for_pred(sample)}{self.verbalizer[label]}"


class SST2Template(ClassificationTemplate):
    verbalizer = {0: " bad", 1: " good"}

    def verbalize_for_pred(self, sample):
        text = sample["sentence"].strip()
        return f"{text} It was"

    def verbalize(self, sample):
        label = sample["label"]
        return f"{self.verbalize_for_pred(sample)}{self.verbalizer[label]}"


class BoolQTemplate(ClassificationTemplate):
    verbalizer = {0: "No", 1: "Yes"}

    def verbalize_for_pred(self, sample):
        passage = sample["passage"]
        question = sample["question"]
        return f"{passage}\nQuestion: {question}\nIs that correct? Yes or No?\n"

    def verbalize(self, sample):
        label = sample["label"]
        return f"{self.verbalize_for_pred(sample)}{self.verbalizer[label]}"


class RTETemplate(ClassificationTemplate):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No"}

    def verbalize_for_pred(self, sample):
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        return f'{premise}\nDoes this mean that "{hypothesis}" is true? Yes or No?\n'


class MultiRCTemplate(ClassificationTemplate):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def verbalize_for_pred(self, sample):
        paragraph = sample["paragraph"]
        question = sample["question"]
        answer = sample["answer"]
        return f'{paragraph}\nQuestion: {question}\nI found this answer "{answer}". Is that correct? Yes or No?\n'


class CBTemplate(ClassificationTemplate):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def verbalize_for_pred(self, sample):
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        return (
            f'Suppose {premise} Can we infer that "{hypothesis}"? Yes, No, or Maybe?\n'
        )


class WICTemplate(ClassificationTemplate):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def verbalize_for_pred(self, sample):
        sent1 = sample["sentence1"]
        sent2 = sample["sentence2"]
        word = sample["word"]
        return f'Does the word "{word}" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n'


class WSCTemplate(ClassificationTemplate):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def verbalize_for_pred(self, sample):
        text = sample["text"]
        span1 = sample["span1_text"]
        span2 = sample["span2_text"]
        return f'{text}\nIn the previous sentence, does the pronoun "{span2.lower()}" refer to {span1}? Yes or No?\n'


class SQuADTemplate(Template):
    def encode(self, sample):
        question = sample["question"].strip()
        title = sample["title"]
        context = sample["context"]

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample):
        question = sample["question"].strip()
        title = sample["title"]
        context = sample["context"]
        answer = sample["answers"][
            0
        ]  # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer: {answer}\n"


class DROPTemplate(Template):
    def encode(self, sample):
        question = sample["question"].strip()
        context = sample["context"]
        return f"Passage: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample):
        question = sample["question"].strip()
        context = sample["context"]
        answer = sample["answers"][
            0
        ]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer: {answer}\n"


class LmTask(Enum):
    sst2 = "sst2"
    rte = "rte"
    multirc = "multirc"
    cb = "cb"
    wic = "wic"
    wsc = "wsc"
    boolq = "boolq"
    squad = "squad"
    drop = "drop"


LM_DATASET_MAP = {
    LmTask.sst2.name: "glue",
    LmTask.rte.name: "super_glue",
    LmTask.multirc.name: "super_glue",
    LmTask.cb.name: "super_glue",
    LmTask.wic.name: "super_glue",
    LmTask.wsc.name: "super_glue",
    LmTask.boolq.name: "super_glue",
    LmTask.squad.name: "squad",
    LmTask.drop.name: "drop",
}

LM_TEMPLATE_MAP = {
    LmTask.sst2.name: SST2Template,
    LmTask.rte.name: RTETemplate,
    LmTask.multirc.name: MultiRCTemplate,
    LmTask.cb.name: CBTemplate,
    LmTask.wic.name: WICTemplate,
    LmTask.wsc.name: WSCTemplate,
    LmTask.boolq.name: BoolQTemplate,
    LmTask.squad.name: SQuADTemplate,
    LmTask.drop.name: DROPTemplate,
}


@dataclass
class LLMBatchInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device=None, dtype=None):
        self.input_ids = self.input_ids.to(device=device)
        self.attention_mask = self.attention_mask.to(device=device)
        return self


def get_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        # Pad sequences to the max length in the batch
        padded_batch = tokenizer.pad(
            {"input_ids": batch},
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = padded_batch["input_ids"]
        attention_mask = padded_batch["attention_mask"]
        return (
            LLMBatchInput(input_ids[:, :(-1)], attention_mask[:, :(-1)]),
            input_ids[:, 1:],
        )  # Prepare input and target sequences

    return collate_fn


# This is used for SQuAD dataset
def get_collate_fn_for_gen_model(tokenizer, max_length):
    def collate_fn(batch):
        inputs, golds = zip(*batch)
        # Pad sequences to the max length in the batch
        padded_batch = tokenizer.pad(
            {"input_ids": inputs},
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = padded_batch["input_ids"]
        attention_mask = padded_batch["attention_mask"]
        return (LLMBatchInput(input_ids, attention_mask), golds)

    return collate_fn


def get_lm_loss(
    loss_type: Literal["full_sentence", "last_token", "accuracy", "f1"],
    verbalizer_id_map: dict[int, int] | None,
):

    if loss_type == "f1":
        return f1_loss

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
    last_token_batch_pred = logits[:, -1, verbalizer_id_list].view(
        -1, len(verbalizer_id_list)
    )
    last_token_label = (sentence_label_tokens[:, -1] == verbalizer_id_map[1]).to(int)

    loss = torch.nn.functional.cross_entropy(last_token_batch_pred, last_token_label)
    return loss


def last_token_accuracy(
    batch_pred, sentence_label_tokens, verbalizer_id_map, verbalizer_id_list
):
    logits = batch_pred.logits
    last_token_batch_pred = logits[:, -1, verbalizer_id_list].view(
        -1, len(verbalizer_id_list)
    )
    last_token_label = (sentence_label_tokens[:, -1] == verbalizer_id_map[1]).to(int)

    pred = last_token_batch_pred.max(1, keepdim=True)[1]
    return pred.eq(last_token_label.view_as(pred)).cpu().float().mean()


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(pred: str, gold: list[str]) -> float:
    if gold[0] == "CANNOTANSWER" or gold[0] == "no answer":
        return int(normalize_answer(gold[0]) == normalize_answer(pred))
    else:
        all_f1s = []
        for ans in gold:
            prediction_tokens = normalize_answer(pred).split()
            ground_truth_tokens = normalize_answer(ans).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                all_f1s.append(0)
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                all_f1s.append((2 * precision * recall) / (precision + recall))
        return np.max(all_f1s)


def f1_loss(
    batch_pred: torch.Tensor, golden_outputs: Sequence[tuple[int, str]], tokenizer
) -> torch.Tensor:
    assert batch_pred.shape[0] == len(golden_outputs)
    f1s = []
    for pred, pos_and_gold in zip(batch_pred, golden_outputs):
        start_pos, gold_sentence = pos_and_gold
        pred_stence = tokenizer.decode(
            pred[start_pos:], skip_special_tokens=True
        ).strip()
        f1s.append(pred_stence, gold_sentence)
    return -torch.tensor(np.mean(f1s), dtype=torch.float32)
