from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_chatbot():
    model = AutoModelForCausalLM.from_pretrained("./chatbot_model")
    tokenizer = AutoTokenizer.from_pretrained("./chatbot_model")
    return model, tokenizer

def generate_response(model, tokenizer, user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response
