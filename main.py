import platform
import os
import random
from torch.utils.data import DataLoader
import os
import torch
# from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
    default_data_collator,
    get_cosine_schedule_with_warmup, 
    GenerationConfig

)
from trl import SFTTrainer

from datapreprocessing import  print_data, print_prompt
from tokenization import  json_open,  create_prompt, pack,token_encode
from dataloader import loader, print_loader
from model import quant4bit_config, training_params
 


def main():

    KOALPACA_PATH = '../dataset/ko_alpaca_data.json'  # 49620
    ALPACA_PATH = "../dataset/alpaca_gpt4_data.json"  # 52002
    MODEL_ID = '../llama_model/llama-2-7b-hf'      # base model
    # MODEL_ID = '../llama_model/llama-2-7b-chat-hf'   # base model
    NEW_MODEL = './model/llama_2_7b_chat_hf'    # fine-tuned model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    ##### 0.data load
    ### data.keys() = ['instruction', 'input', 'output']
    alpaca = json_open(ALPACA_PATH)
    num_data = 230
    # print_data(alpaca, num_data )

    ##### 1. Dataset preparation and tokenization
    ### prompts = 특정 문구, 'instruction', 'Response'  if is not 'input'
    ### prompts = 특정 문구, 'instruction', 'input',  'Response'  if is 'input'
    prompts = [create_prompt(row) for row in alpaca]
    # print_prompt(alpaca, data_num)

    ##### 1.1. End of String Token (EOS)
    #### output의 끝 표시
    EOS_TOKEN = "</s>"
    ### outputs :  output + </s>
    outputs = [row['output'] + EOS_TOKEN for row in alpaca]

    
    dataset = [{"prompt":s, "output":t, "example": s+t} for s, t in zip(prompts, outputs)]

    ##### 1.2. tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                            trust_remote_code=True, # 자체 모델링 파일에서 허브에 정의된 사용자 정의 모델을 허용할지 여부
                                            padding=True
                                        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'   # default
    # print('tokenizer:', token_encode(tokenizer))
    
    # 1.3. Creating a Train-Eval Split
    random.shuffle(dataset)
    train_dataset = dataset[:-1000]
    eval_dataset = dataset[-1000:]

    max_seq_len = 1024

    # 1.4. concatenate a bunch of separated by the EOS token together
    train_ds_packed = pack(tokenizer, train_dataset, max_seq_len)  # 11266
    eval_ds_packed = pack(tokenizer, eval_dataset, max_seq_len)    # 215

    print('train_ds_packed: ', len(train_dataset), '->' ,len(train_ds_packed))
    print('eval_ds_packed: ', len(eval_dataset), '->', len(eval_ds_packed))
    print('train_ds_packed.keys()', train_ds_packed[0].keys())


   
    # 2. DataLoader
    batch_size= 8
    train_dataloader = loader(train_ds_packed, batch_size, 'train')
    eval_dataloader  = loader(eval_ds_packed, batch_size, 'eval')
    # print_loader(train_dataloader)

    # 3. Training Loop
    # 3.1. model
    # config = get_config(train_dataloader, max_seq_len, batch_size, MODEL_ID)
    training_paramas = training_params()
    quant_config = quant4bit_config()
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map={"": 0}
            # device_map=0,
            # trust_remote_code=True,
            # low_cpu_mem_usage=True,
            # torch_dtype=torch.bfloat16,
            # use_cache=False,
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print(model)
    raise ValueError
    # 3.2. train a subset of the model parameter
    # Llama 2-7b has 32 transformer layers
    n_freeze = 24.  # train the last 8 of 32 transformer layers

    # 3.2.1. freeze layers (disable gradients) -> Gain a ton of memory by not computing gradients
    for param in model.parameters(): param.requires_grad = False
    for param in model.lm_head.parameters(): param.requires_grad = True
    for param in model.model.layers[n_freeze:].parameters(): param.requires_grad = True

    # 3.2.2. freezing the embeddings-> gain a little bit more memory
    if config.freeze_embed:
        model.model.embed_tokens.weight.requires_grad_(False)

    # 3.3.3. gradient checkpointing ->save more memory, but training slower
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})



    # 4. Optimizer and Scheduler
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9,0.99), eps=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_training_steps=config.total_train_steps,  # The number of steps for the warmup phase.
        num_warmup_steps=config.total_train_steps // 10,  # The total number of training steps.
    )

    # 5. loss
    def loss_fn(x, y):
        "A Flat CrossEntropy" 
        return torch.nn.functional.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))


    # 6. model output print
    gen_config = GenerationConfig.from_pretrained(config.model_id)
    prompt = " i love you"
    decode_output = generate(model, tokenizer, prompt, max_new_tokens=100, gen_config=gen_config)
    print('decode_output', decode_output)

    # 7.Validation Step
    def validate():
        model.eval();
        eval_acc = Accuracy()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = to_gpu(batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = loss_fn(out.logits, batch["labels"])
                eval_acc.update(out.logits, batch["labels"])
        prompt_table(eval_dataset[:config.n_eval_samples], log=True)
        model.train();

    # 8. A Simple PyTorch Training Loop for Your LLM
    # Training
    acc = Accuracy()
    model.train()
    train_step = 0
    pbar = tqdm(total=config.total_train_steps)
    for epoch in range(config.epochs):
        for step, batch in enumerate(train_dataloader):
            batch = to_gpu(batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = loss_fn(out.logits, batch["labels"]) / config.gradient_accumulation_steps  # you could use out.loss and not shift the dataset  
                loss.backward()
            if step%config.gradient_accumulation_steps == 0:
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                train_step += 1
                pbar.update(1)
        validate()
    pbar.close()
    # we save the model checkpoint at the end
    save_model(
        model, 
        model_name=config.model_id.replace("/", "_"), 
        models_folder="models/", log=config.log_model)
        



    















if __name__ == "__main__":
    # token = "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" 
    # login(token)
    main()