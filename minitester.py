import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the trained model and tokenizer
model_path = "C:/Users/Sagar/Documents/llm-vscode-project/trainedmodels"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Function to generate text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example prompt
prompt = "A car has"
generated_text = generate_text(prompt)

print("Prompt: ", prompt)
print("Generated Text: ", generated_text)
