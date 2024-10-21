import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

# # Uncomment here if got error of importing flood_detection module
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flood_detection.model import FloodDetectionModel
from flood_detection.data_preprocessing import get_data_transforms
from risk_assessment.risk_classifier import assess_risk

# Import chatbot modules
from chatbot.predict import load_chatbot, generate_response
# Load the chatbot model and tokenizer
chatbot_model, chatbot_tokenizer = load_chatbot()

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flood_model = FloodDetectionModel().to(device)
flood_model.load_state_dict(torch.load("flood_detection/checkpoints/20241020_213048_last.pt", map_location=device)["model_state_dict"])
flood_model.eval()

chatbot_model, chatbot_tokenizer = None, None  # load_chatbot()

st.title("AI-Powered Flood Detection and Response System")

uploaded_file = st.file_uploader("Choose an image for flood detection", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    transform = get_data_transforms()['val']
    input_tensor = transform(np.array(image)).unsqueeze(0).to(device)

    # Perform flood detection
    with torch.no_grad():
        output = flood_model(input_tensor)
        flood_prob = F.softmax(output).squeeze()
        # flood_mask = (output > 0.5).squeeze().cpu().numpy()

    # Assess risk
    risk_level, flood_percentage = "???", flood_prob[1]  # assess_risk(flood_mask)

    st.write(f"Flood Percentage: {flood_percentage:.2%}")
    st.write(f"Risk Level: {risk_level}")

    # Display flood mask
    # st.image(flood_mask, caption="Flood Mask", use_column_width=True, clamp=True)

st.subheader("Flood Response Chatbot")
user_input = st.text_input("Ask a question about flood response:")
if user_input:
    response = generate_response(chatbot_model, chatbot_tokenizer, user_input)
    st.write(f"Chatbot: {response}")
