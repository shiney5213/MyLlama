import platform
import os
import random
from torch.utils.data import DataLoader


from tokenization import json_open, create_prompt, pack
from dataloader import loader, loadercheck
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    default_data_collator
)
from peft import LoraConfig
from trl import SFTTrainer

def main():

    KOALPACA_PATH = '../dataset/ko_alpaca_data.json'  # 49620
    ALPACA_PATH = "../dataset/alpaca_gpt4_data.json"  # 52002
    MODEL_ID = '../llama_model/llama-2-7b-hf'


    # 0.data load
    alpaca = json_open(ALPACA_PATH)

    row_num = 232
    row = alpaca[row_num]
    # print(prompt_input(row))


    # 1. Dataset preparation and tokenization
    prompts = [create_prompt(row) for row in alpaca]  # all LLM inputs are here
    print('prompts: ', prompts[row_num])


    # 1.1. End of String Token (EOS)
    # output의 끝을 표시
    EOS_TOKEN = "</s>"
    outputs = [row['output'] + EOS_TOKEN for row in alpaca]
    print('outputs:', outputs[row_num])

    dataset = [{"prompt":s, "output":t, "example": s+t} for s, t in zip(prompts, outputs)]


    # 1.2. Tokens, tokens everywhere: How to tokenize and organize text
    # 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                            trust_remote_code=True, # 자체 모델링 파일에서 허브에 정의된 사용자 정의 모델을 허용할지 여부
                                            padding=True
                                        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'   # default



    # text is sentence
    test_text = "My experiments are going strong!"

    # 1.2.1. pad to length of longest
    padding = 'longest'

    # 1.2.2. pad to length of max length
    padding = 'max_length'
    # max_length = 10

    longest_token = tokenizer.encode(test_text, 
                                padding='longest',
                                return_tensors="pt"   # type : tensor
                                )
    max_length_token = tokenizer.encode(test_text, 
                                padding='max_length',
                                max_length = 10,
                                return_tensors="pt"   # type : tensor
                                )                            
    print('padding: longest', longest_token)
    print('padding: max_length', max_length_token)


    # 1.3. Creating a Train-Eval Split
    random.shuffle(dataset)
    train_dataset = dataset[:-1000]
    eval_dataset = dataset[-1000:]

    max_seq_len = 1024

    # 1.4. concatenate a bunch of separated by the EOS token together
    train_ds_packed = pack(tokenizer, train_dataset, max_seq_len)  # 11266
    eval_ds_packed = pack(tokenizer, eval_dataset, max_seq_len)    # 215

    print('train_ds_packed: ', len(train_dataset), '->' ,len(train_ds_packed))
    print('train_ds_packed[0]', train_ds_packed[0].keys())
    print('eval_ds_packed: ', len(eval_dataset), '->', len(eval_ds_packed))


    # 1.5. Batching multiple sequences of different lengths-> text is list
    # 1.5.1. pad to length of longest
    test_text = ["My experiments are going strong!", "I love Llamas"]

    padding = 'longest'
    longest_token = tokenizer(test_text, 
                                padding=padding,
                                return_tensors="pt"   # type : tensor
                                )

    # 1.5.2. pad to length of max length
    padding = 'max_length'
    max_length = 10

    max_length_token = tokenizer(test_text, 
                                padding=padding,
                                max_length = max_length,
                                return_tensors="pt"   # type : tensor
                                )                            
    print('padding: longest', longest_token)
    print('padding: max_length', max_length_token)


    # 2. DataLoader
    batch_size= 8
    train_dataloader = loader(train_ds_packed, batch_size, 'train')
    eval_dataloader  = loader(eval_ds_packed, batch_size, 'eval')

    loadercheck(train_dataloader)

    # 3. Training Loop













if __name__ == "__main__":
    main()