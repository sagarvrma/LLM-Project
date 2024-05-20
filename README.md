# LLM-Project
**README.md**

### Overview
This repository contains code for training a GPT-2 language model using a smaller subset of the WikiText-2 dataset and then using the trained model for text generation. The training process is implemented in `train_llm_mini.py`, and the text generation is demonstrated in `minitester.py`.

### Dependencies
- Python 3.6 or higher
- PyTorch
- Transformers library
- Datasets library

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/llm-vscode-project.git
   cd llm-vscode-project
   ```

2. Install the required libraries:
   ```bash
   pip install torch transformers datasets
   ```

### Usage
1. **Training the Model (`train_llm_mini.py`):**
   - This script loads the WikiText-2 dataset and uses a smaller subset of 1000 samples for training, as the full dataset from huggingface was taking about 35 hours because I do not have a powerful GPU.
   - It initializes the GPT-2 tokenizer and model, and then tokenizes the dataset.
   - Training arguments such as output directory, number of epochs, batch size, and logging steps are defined.
   - The model is trained using the Trainer class from the Transformers library.
   - The trained model is saved to the specified directory.

   To train the model, run:
   ```bash
   python train_llm_mini.py
   ```

2. **Text Generation (`minitester.py`):**
   - This script loads the trained GPT-2 model and tokenizer.
   - It sets the model to evaluation mode and defines a function for text generation.
   - A prompt is provided, and the model generates text based on the prompt.

   To generate text, run:
   ```bash
   python minitester.py
   ```

### Relationship between Files
- `train_llm_mini.py` is used to train the GPT-2 language model using a smaller subset of the WikiText-2 dataset. It saves the trained model to a specified directory.
- `minitester.py` loads the trained model and tokenizer to generate text based on a prompt. It demonstrates the use of the trained model for text generation.

### Note
- Adjust the training arguments in `train_llm_mini.py` according to your computing resources and training requirements.
- The generated text in `minitester.py` may vary based on the prompt and the training data used.
