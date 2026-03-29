import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Optimize memory allocation for Laptop GPUS
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv()
token = os.getenv("HF_TOKEN")
if token and token != "your_token_here":
    login(token=token)
else:
    print("⚠️ WARNING: No HF_TOKEN found in .env. Llama 3.1 download will fail.")

# --- Hardware Check ---
print("==================================================")
device_type = "GPU" if torch.cuda.is_available() else "CPU"
print(f"🚀 INITIALIZING TRAINING ON: {device_type}")
if device_type == "GPU":
    print(f"💻 DETECTED DEVICE: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: No GPU detected! Training will be extremely slow.")
print("==================================================")

from trl import SFTTrainer, SFTConfig

# 1. Configuration
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
train_file = "train_sft.jsonl"
eval_file = "eval_sft.jsonl"
output_dir = "./llama-sft-model"

# 2. Load Model & Tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Llama 3 requires specific chat template handling
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False 

# 3. Prepare for LoRA (Target all linears for QLoRA)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# CRITICAL: For standard Trainer, we MUST explicitly wrap the model with PEFT
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Load & Tokenize Datasets
print("Loading SFT datasets...")
raw_dataset = load_dataset("json", data_files={"train": train_file, "test": eval_file})

def tokenize_function(examples):
    # Apply Chat Template AND Tokenize
    texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples["messages"]]
    return tokenizer(
        texts,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

print("Tokenizing dataset...")
tokenized_datasets = raw_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=raw_dataset["train"].column_names
)

# 5. Training with Standard Trainer (Version-Agnostic)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1, # Memory: process one eval sample at a time
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,    # Memory: offload eval results to CPU to save VRAM
    warmup_steps=100, 
    num_train_epochs=3, 
    learning_rate=2e-5,
    bf16=True,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("Starting training run...")
trainer.train()

# 6. Save Adapter
model.save_pretrained(output_dir)
print(f"SFT Model adapter saved to {output_dir}")
