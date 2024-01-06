import json


def json_open(JSON_PATH):
    
    with open(JSON_PATH, 'r', encoding = 'utf8') as f:
        alpaca = f.read()

    alpaca = json.loads(alpaca)
    print('data len', len(alpaca) )

    return alpaca


def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)
    
def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)
    

def create_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)


def pack(tokenizer, dataset, max_seq_len=1024):
    tkds_ids = tokenizer([s["example"] for s in dataset])["input_ids"]
    
    all_token_ids = []
    for tokenized_input in tkds_ids:
        all_token_ids.extend(tokenized_input + [tokenizer.eos_token_id])
    
    packed_ds = []
    for i in range(0, len(all_token_ids), max_seq_len+1):
        input_ids = all_token_ids[i : i + max_seq_len+1]
        if len(input_ids) == (max_seq_len+1):
            # as input and target are the same but shifted, we lose one token at each end
            packed_ds.append({"input_ids": input_ids[:-1], "labels": input_ids[1:]})  # < --- ‼️ ⛔️
	    # if you use the model.output.loss you don't need to shift, it is done for you!
    return packed_ds











