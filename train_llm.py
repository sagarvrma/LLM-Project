import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./trainedmodels',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("C:/Users/Sagar/Documents/llm-vscode-project/trainedmodels")
