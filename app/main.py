import streamlit as st
from PIL import Image
import torch
import numpy as np
from flood_detection.model import FloodDetectionModel
from flood_detection.data_preprocessing import get_data_transforms
# from risk_assessment.risk_classifier import assess_risk
# from chatbot.predict import load_chatbot, generate_response

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flood_model = FloodDetectionModel().to(device)
# flood_model.load_state_dict(torch.load('flood_detection_model.pth', map_location=device))
flood_model.eval()

chatbot_model, chatbot_tokenizer = 0, 0  # load_chatbot()

st.title("AI-Powered Flood Detection and Response System")

uploaded_file = st.file_uploader("Choose an image for flood detection", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    transform = get_data_transforms()['val']
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Perform flood detection
    with torch.no_grad():
        output = flood_model(input_tensor)
        flood_mask = (output > 0.5).squeeze().cpu().numpy()

    # Assess risk
    risk_level, flood_percentage = 0, 0  # assess_risk(flood_mask)

    st.write(f"Flood Percentage: {flood_percentage:.2%}")
    st.write(f"Risk Level: {risk_level}")

    # Display flood mask
    st.image(flood_mask, caption="Flood Mask", use_column_width=True, clamp=True)

st.subheader("Flood Response Chatbot")
user_input = st.text_input("Ask a question about flood response:")
# if user_input:
#     response = generate_response(chatbot_model, chatbot_tokenizer, user_input)
#     st.write(f"Chatbot: {response}")
