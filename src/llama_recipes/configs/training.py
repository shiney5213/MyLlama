
from dataclasses import dataclass
import torch



@dataclass
class train_config:

    ##### basic parameter
    seed: int=42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CUDA_VISIBLE_DEVICES = "0"

    ### path
    # output_dir: str = "PATH/to/save/PEFT/model"
    output_dir: str = "results/model"

    ### dataset
    # dataset = "samsum_dataset"
    dataset = "alpaca_dataset"
    # dataset = "grammar_dataset"
    # dataset = "custom_dataset"

    ### model
    model_name : str= '../../../llama_models/llama-2-7b-hf' 

    ### save
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    
    ### basic setting
    # num_epochs: int=3
    num_epochs: int=1
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    mixed_precision: bool=True
    val_batch_size: int=1

    run_validation: bool=True
    # batch_size_training: int=4
    batch_size_training: int=1
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    

    
    
    ##### setting
    one_gpu: bool = False

    # FSDP(Fully Sharded Data Parallel)
    # enable_fsdp: bool=False   
    enable_fsdp: bool=True   
    low_cpu_fsdp: bool=False   # 70B 모델 조정할 때 사용

    # PEFT(Parameter Efficient Fine-tuning)
    use_peft: bool=False
    peft_method: str = "lora" # None , llama_adapter, prefix



    # Create a gradient scaler for fp16
    use_fp16: bool=False
    # enable_fsdp: bool=False 

    # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_fast_kernels: bool = False 

    # quantization: bool = False
    quantization: bool = True

   
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
               
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    
    



# from types import SimpleNamespace


# gradient_accumulation_steps = 32 // batch_size


# config = SimpleNamespace(
#     model_id='meta-llama/Llama-2-7b-hf',
#     dataset_name="alpaca-gpt4",
#     precision="bf16",  # faster and better than fp16, requires new GPUs
#     n_freeze=24,  # How many layers we don't train, LLama 7B has 32.
#     lr=2e-4,
#     n_eval_samples=10, # How many samples to generate on validation
#     max_seq_len=max_sequence_len, # Length of the sequences to pack
#     epochs=3,  # we do 3 pasess over the dataset.
#     gradient_accumulation_steps=gradient_accumulation_steps,  # evey how many iterations we update the gradients, simulates larger batch sizes
#     batch_size=batch_size,  # what my GPU can handle, depends on how many layers are we training  
#     log_model=False,  # upload the model to W&B?
#     mom=0.9, # optim param
#     gradient_checkpointing = True,  # saves even more memory
#     freeze_embed = True,  # why train this? let's keep them frozen ❄️
# )


# config.total_train_steps = config.epochs * len(train_dataloader) // config.gradient_accumulation_steps
