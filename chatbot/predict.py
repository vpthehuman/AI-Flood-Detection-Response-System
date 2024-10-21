import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to load the fine-tuned chatbot model and tokenizer
def load_chatbot():
    model_path = "./chatbot_model/flood-dialoGPT"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Function to generate a response (Adjustable Hyperparameters)
def generate_response(model, tokenizer, prompt):  # Controls response length; increase for longer outputs
    # Tokenize the input prompt
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt", padding=True)

    # Create attention mask explicitly
    attention_mask = inputs["attention_mask"]
    
    # Generate a response using the model
    output = model.generate(
        inputs["input_ids"],
        attention_mask=attention_mask,  # Adding attention_mask to avoid the warning
        max_length=100,          # Maximum length of response
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,  # Avoid repeating phrases
        do_sample=True,          # Enable sampling to get more diverse outputs
        top_k=50,                # Limit to top-k sampling; can be adjusted for variation
        top_p=0.95,              # Nucleus sampling; can be adjusted for variation
        temperature=0.7          # Adjust randomness; lower for more deterministic outputs
        # Suggested Values:
        # - `max_length`: Increase to 150-200 if you need longer responses.
        # - `top_k` and `top_p`: Adjust for more or less diversity. Lower `top_k` or `top_p` for more focused answers.
        # - `temperature`: Lower to 0.5 for more deterministic and less creative responses; increase to 1.0 for more varied responses.
    )
    
    # Decode the generated tokens
    response = tokenizer.decode(output[:, inputs["input_ids"].shape[-1]:][0], skip_special_tokens=True)
    return response

# Function to process multiple user inputs from a JSON file
def process_user_inputs(model, tokenizer, user_input_file="./chatbot/user_input.json"):
    with open(user_input_file, "r") as file:
        user_inputs = json.load(file)

    responses = []
    # Loop through all user inputs and generate responses
    for user_input in user_inputs:
        response = generate_response(model, tokenizer, user_input["input"])  # Assuming "input" key in JSON file
        responses.append({"User": user_input["input"], "Bot": response})
    return responses

# Example usage if run directly
if __name__ == "__main__":
    model, tokenizer = load_chatbot()
    responses = process_user_inputs(model, tokenizer)
    for conversation in responses:
        print(f"User: {conversation['User']}")
        print(f"Bot: {conversation['Bot']}\n")
