from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model
)
import torch

# -----------------------------------
# MODEL NAME
# -----------------------------------
model_name = "mistralai/Mistral-7B-v0.1"

# -----------------------------------
# QUANTIZATION CONFIG
# -----------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# -----------------------------------
# LOAD TOKENIZER
# -----------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# -----------------------------------
# LORA CONFIG
# -----------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# -----------------------------------
# APPLY PEFT
# -----------------------------------
model = get_peft_model(model, lora_config)

# -----------------------------------
# LOAD DATASET
# -----------------------------------
dataset = load_dataset(
    "json",
    data_files=["finance_dataset.json", "tech_dataset.json"]
)

# -----------------------------------
# FORMAT DATA
# -----------------------------------
def format_data(example):
    text = f"""
Instruction:
{example['instruction']}
Response:
{example['response']}
"""
    return {"text": text}

dataset = dataset.map(format_data)

# -----------------------------------
# TOKENIZE
# -----------------------------------
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize)

# -----------------------------------
# TRAINING ARGUMENTS
# -----------------------------------
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True
)

# -----------------------------------
# TRAINER
# -----------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# -----------------------------------
# START TRAINING
# -----------------------------------
trainer.train()

# -----------------------------------
# SAVE MODEL
# -----------------------------------
trainer.model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Training Completed")
