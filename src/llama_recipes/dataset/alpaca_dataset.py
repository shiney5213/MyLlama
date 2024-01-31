# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset


# PROMPT_DICT = {
#     "prompt_input": (
<<<<<<< HEAD
#         "Below is an instruction that describes a task, paired with an input that provides further context.\n"
#         "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
#         "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
#         "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task.\n"
#         "아래는 작업을 설명하는 명령어입니다.\n\n"
#         "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
#         "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
=======
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
>>>>>>> 2b37f4108afc35b18939596cb88db459af30545e
#     ),
# }

PROMPT_DICT = {
    "prompt_input": (
<<<<<<< HEAD
        "다음은 질문과 함께 제공되는 입력을 만족하는 답변을 하는 예제입니다.\n\n"
        "질문에 대한 답변을 작성하세요.\n\n"
        "### 질문:\n{instruction}\n\n### 입력:\n{input}\n\n### 답변:"
    ),
    "prompt_no_input": (
        "다음 질문에 대한 답변을 작성하세요.\n\n"
        "### 질문:\n{instruction}\n\n### 답변:"
=======
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
        "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n"
        "아래는 작업을 설명하는 명령어입니다.\n\n"
        "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
>>>>>>> 2b37f4108afc35b18939596cb88db459af30545e
    ),
}



class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        try:
            self.ann = json.load(open(dataset_config.data_path))
        except: 
            self.ann = json.load(open(dataset_config.data_path, 'rt', encoding = 'utf-8-sig') )
        if partition == "train":
            self.ann = self.ann[200:]
        else:
            self.ann = self.ann[:200]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
