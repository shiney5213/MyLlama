
from transformers import  BitsAndBytesConfig, TrainingArguments
import torch
from types import SimpleNamespace
from peft import LoraConfig


def quant4bit_config():
    """4-bit quantization with NF4 type configuration using BitsAndBytes
    representing weights and activations with lower-precision data types like 4-bit integers (int4)
    -> reduces memory and computational costs
    -> speeding up inference
    """

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # 4bit로 데이터 로드
        bnb_4bit_quant_type="nf4",               # 양자화 유형: nf4
        bnb_4bit_compute_dtype=compute_dtype,   # compute_dtype : float 16
        bnb_4bit_use_double_quant=False,        # 이중 양자화 사용 여부
    )

    return quant_config

    
def get_config(train_dataloader, max_sequence_len, batch_size, MODEL_ID):
    # 일반적인 방법
    gradient_accumulation_steps = 32 // batch_size


    config = SimpleNamespace(
        model_id= MODEL_ID, 
        dataset_name="alpaca-gpt4",
        precision="bf16",  # faster and better than fp16, requires new GPUs
        n_freeze=24,  # How many layers we don't train, LLama 7B has 32.
        lr=2e-4,
        n_eval_samples=10, # How many samples to generate on validation
        max_seq_len=max_sequence_len, # Length of the sequences to pack
        epochs=3,  # we do 3 pasess over the dataset.
        gradient_accumulation_steps=gradient_accumulation_steps,  # evey how many iterations we update the gradients, simulates larger batch sizes
        batch_size=batch_size,  # what my GPU can handle, depends on how many layers are we training  
        log_model=False,  # upload the model to W&B?
        mom=0.9, # optim param
        gradient_checkpointing = True,  # saves even more memory
        freeze_embed = True,  # why train this? let's keep them frozen ❄️
    )


    config.total_train_steps = config.epochs * len(train_dataloader) // config.gradient_accumulation_steps

    return config

def training_params():
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    return training_params

def peft_params():
    """
    update all of the model's parameters -> computationally expensive and requires massive amounts of data.
    Parameter-Efficient Fine-Tuning (PEFT)
    : only update a small subset of the model's parameters -> much more efficient
    """
    peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


    return peft_params

