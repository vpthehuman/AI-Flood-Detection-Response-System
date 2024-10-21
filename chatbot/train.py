import os
import json  # Import JSON module for reading the conversation file
os.environ["USE_TF"] = "0"  # Ensure transformers uses PyTorch only

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

def train_chatbot():
    # Step 0: Ensure the output and logging directories exist
    output_dir = "./chatbot_model/flood-dialoGPT"
    logging_dir = './chatbot_model/logs'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Step 1: Load Sample Conversations from JSON File
    conversation_file = "./chatbot/conversations.json"  # Specify your JSON file name
    with open(conversation_file, "r") as file:
        conversations = json.load(file)

    # Step 2: Create a Custom Dataset Class
    class FloodDataset(Dataset):
        def __init__(self, conversations, tokenizer, max_length=128):
            self.tokenizer = tokenizer
            self.conversations = conversations
            self.max_length = max_length

        def __len__(self):
            return len(self.conversations)

        def __getitem__(self, idx):
            input_text = self.conversations[idx]["input"]
            response_text = self.conversations[idx]["response"]
            combined = f"{input_text} [SEP] {response_text}"

            # Tokenize and encode the text
            encoded = self.tokenizer.encode_plus(
                combined,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].squeeze()
            attention_mask = encoded["attention_mask"].squeeze()

            # To calculate loss, we need labels; the labels will be the same as the input_ids
            labels = input_ids.clone()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    # Step 3: Load Model and Tokenizer
    model_name = "microsoft/DialoGPT-medium"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add a padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 4: Prepare the Dataset
    train_dataset = FloodDataset(conversations, tokenizer)

    # Step 5: Set Training Arguments (Adjustable Hyperparameters)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Controls batch size per device; adjust for larger datasets
        num_train_epochs=1,             # Number of epochs; more epochs for better learning on larger datasets
        save_steps=10,                  # How frequently checkpoints are saved
        save_total_limit=2,             # Maximum number of checkpoints to keep
        logging_dir=logging_dir,
        logging_steps=10,               # How frequently logs are printed during training
        # Suggested Values:
        # - `per_device_train_batch_size`: Try values like 4, 8, or 16 based on your GPU memory.
        # - `num_train_epochs`: For larger datasets, try 3-5 epochs or more for better convergence.
        # - `save_steps`: Increase to 100 or 1000 for longer training to save checkpoints less frequently.
        # - `temperature` (in `Prediction.py`): Decrease (e.g., 0.5) for more predictable responses, increase (e.g., 1.0) for more creative responses.
    )

    # Step 6: Initialize Trainer and Start Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    # Step 7: Save the Model and Tokenizer
    trainer.save_model(output_dir)  # Save the final model
    tokenizer.save_pretrained(output_dir)

# Main function to execute the chatbot training
if __name__ == "__main__":
    train_chatbot()
