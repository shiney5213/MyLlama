import json
from tokenization import create_prompt

def print_data(dataset, num_data):
    print(dataset[num_data])



def print_prompt(alpaca, row_num):
    row = alpaca[row_num]
    # print(prompt_input(row))

    prompts = [create_prompt(row) for row in alpaca]
    # 1. Dataset preparation and tokenization
    print(f'{row_num}th prompt: ')
    print( prompts[row_num])

    

    