import streamlit as st
from PIL import Image
import torch
from flood_detection.model import FloodDetectionModel
from flood_detection.data_preprocessing import get_data_transforms
from risk_assessment.risk_classifier import assess_risk
from chatbot.predict import load_chatbot, generate_response

# Load models
flood_model = FloodDetectionModel()
flood_model.load_state_dict(torch.load('flood_detection_model.pth'))
flood_model.eval()

chatbot_model, chatbot_tokenizer = load_chatbot()

st.title("AI-Powered Flood Detection and Response System")

uploaded_file = st.file_uploader("Choose an image for flood detection", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    transform = get_data_transforms()['val']
    input_tensor = transform(image).unsqueeze(0)
    
    # Perform flood detection
    with torch.no_grad():
        output = flood_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        flood_probability = probabilities[0][1].item()
    
    # For demonstration, we'll use a random value for precipitation intensity
    precipitation_intensity = 0.5  # In a real scenario, this would come from actual data
    
    # Assess risk
    risk_level, risk_score = assess_risk(flood_probability, precipitation_intensity)
    
    st.write(f"Flood Probability: {flood_probability:.2f}")
    st.write(f"Risk Level: {risk_level}")
    st.write(f"Risk Score: {risk_score:.2f}")

st.subheader("Flood Response Chatbot")
user_input = st.text_input("Ask a question about flood response:")
if user_input:
    response = generate_response(chatbot_model, chatbot_tokenizer, user_input)
    st.write(f"Chatbot: {response}")
