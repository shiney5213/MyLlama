from types import SimpleNamespace


gradient_accumulation_steps = 32 // batch_size


config = SimpleNamespace(
    model_id='meta-llama/Llama-2-7b-hf',
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
