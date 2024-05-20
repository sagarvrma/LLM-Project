import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the dataset and use a smaller subset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset = dataset["train"].shuffle(seed=42).select(range(1000))  # Use only 1000 samples

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./trainedmodels',
    num_train_epochs=1,  # Reduce the number of epochs
    per_device_train_batch_size=16,  # Adjust batch size if memory allows
    save_steps=200,  # Save less frequently to reduce overhead
    save_total_limit=2,
    logging_steps=50,  # Log training progress more frequently
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("C:/Users/Sagar/Documents/llm-vscode-project/trainedmodels")
