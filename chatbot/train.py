from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch

def train_chatbot():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load your flood-related dataset here
    # For demonstration, we'll use a small example
    flood_dataset = [
        "What should I do during a flood?",
        "Stay indoors, avoid flooded areas, and listen to emergency services.",
        "How can I prepare for a flood?",
        "Keep emergency supplies, know your evacuation routes, and stay informed.",
    ]

    # Tokenize the dataset
    inputs = tokenizer(flood_dataset, return_tensors="pt", padding=True, truncation=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./chatbot_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs["input_ids"],
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./chatbot_model")
    tokenizer.save_pretrained("./chatbot_model")

if __name__ == "__main__":
    train_chatbot()
