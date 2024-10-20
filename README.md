# AI-Flood-Detection-Response-System
# AI-Powered Flood Detection System with Interactive Chatbot

This project combines advanced deep learning techniques for flood detection with an interactive chatbot to provide a comprehensive flood management solution.

## Features

- Flood detection using ResNet50-based CNN
- Real-time risk assessment
- Interactive chatbot for flood-related queries and guidance
- User-friendly web interface for easy access and visualization

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/AI-Flood-Detection-Response-System.git
    cd AI-Flood-Detection-Response-Sys
    ```


2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use venv\Scripts\activate
    ```


4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```


5. Set up the FloodNet dataset:
    - ~~Place your ColorMasks-FloodNetv1 folder in the `data` directory~~
    - ~~Ensure the subfolders (ColorMask-TestSet, ColorMask-TrainSet, ColorMask-ValSet) are present~~

    Currently, only image classification is available. Download ```data.zip``` from our WhatsApp and unzip it. The result will look like this:

    ```bash
    data
    └── processed
        ├── test
        │   ├── ann
        │   └── img
        ├── train
        │   ├── labeled
        |   │   └── flooded
        |   │       ├── ann
        |   │       └── img
        |   └── non-flooded
        |   |       ├── ann
        |   |       └── img
        |   └── unlabeled
        |       ├── ann
        |       └── img
        └── validation
            ├── ann
            └── img
    ```


5. Train the flood detection model:
    ```bash
    python flood_detection/train_model.py
    ```


6. Fine-tune the chatbot:
    ```bash
    python chatbot/train.py
    ```


7. Run the application:
    ```bash
    streamlit run app/main.py
    ```



## Usage

1. Upload an image for flood detection
2. View the flood detection results and risk assessment
3. Use the chatbot to ask flood-related questions

