import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# Model and training configurations
MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "/content/drive/MyDrive/phi2-qlora-grpo"  # Save to Google Drive
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2  # Reduced batch size for Colab GPU
MAX_LENGTH = 512

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load and prepare the model for QLoRA
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["Wqkv", "out_proj", "fc1", "fc2"]
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Load and preprocess dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

def preprocess_function(examples):
    return tokenizer(
        examples["instruction"] + "\n" + examples["context"] + "\n" + examples["response"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    preprocess_function,
    remove_columns=dataset.column_names,
    batched=True
)

# Configure GRPO training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,  # Increased for stability
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.03,
    logging_steps=5,
    save_strategy="epoch",
    evaluation_strategy="no",
    lr_scheduler_type="cosine",
    report_to="wandb",
    gradient_checkpointing=True,
    fp16=True  # Enable mixed precision training
)

# Configure GRPO
grpo_config = GRPOConfig(
    max_length=MAX_LENGTH,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)

# Initialize GRPO trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    grpo_config=grpo_config
)

# Initialize Weights & Biases for tracking
import wandb
wandb.login()

# Train the model
trainer.train()

# Save the final model
trainer.save_model()
print(f"Model saved to {OUTPUT_DIR}")