# Example usage:
# python merge_peft.py

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import argparse

from dataclasses import dataclass

with open("auth_tokens.json", "r") as f:
	HF_AUTH_TOKEN = json.loads(f.read())["hf_token"]

@dataclass
class config:
    base_model: str='beomi/open-llama-2-ko-7b'
    peft_model: str='./results/model_5th'
    hub_id: str="colable/llama-ko-peft-v0.5"

def main():
    args = config()
    
    print(f"[1/5] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[2/5] Loading adapter: {args.peft_model}")
    model = PeftModel.from_pretrained(base_model, args.peft_model, device_map="auto")
    
    print("[3/5] Merge base model and adapter")
    model = model.merge_and_unload()
    
    print(f"[4/5] Saving model and tokenizer in {args.hub_id}")
    model.save_pretrained(f"{args.hub_id}")
    tokenizer.save_pretrained(f"{args.hub_id}")

    print(f"[5/5] Uploading to Hugging Face Hub: {args.hub_id}")
    model.push_to_hub(f"{args.hub_id}", 
                      use_temp_dir=True, 
			                token=HF_AUTH_TOKEN)
    tokenizer.push_to_hub(f"{args.hub_id}", 
                      use_temp_dir=True, 
			                token=HF_AUTH_TOKEN)
    
    print("Merged model uploaded to Hugging Face Hub!")

if __name__ == "__main__" :
    main()