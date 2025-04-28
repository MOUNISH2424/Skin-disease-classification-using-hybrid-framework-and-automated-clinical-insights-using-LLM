import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model("best_model.keras")

# Define the class mapping
class_map = [
    'BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus',
    'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles'
]

# Load the disease information dataset
disease_info_df = pd.read_csv("data1/final.csv", encoding="ISO-8859-1")

def predict_image(img_path):
    """Predicts the class of the given image."""
    # Load and preprocess the image
    img = load_img(img_path, target_size=(192, 108))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions

    # Make a prediction
    predictions = model.predict([img_array, img_array, img_array])  # Model has 3 inputs
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_disease = class_map[predicted_class]

    return predicted_disease

def get_disease_info(disease_name):
    """Fetches symptoms, causes, and treatment from the dataset."""
    row = disease_info_df[disease_info_df["Disease"] == disease_name]
    if not row.empty:
        return row.iloc[0, 1], row.iloc[0, 2], row.iloc[0, 3]  # Symptoms, Causes, Treatment
    return "No data available", "No data available", "No data available"

# Streamlit UI
st.title("Skin Disease Classification")

st.header("Upload an Image of a Skin Disease")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Save the uploaded image temporarily
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Make the prediction
    predicted_disease = predict_image(img_path)

    # Debugging: print out the predicted disease name to ensure it's correct
    st.write(f"Predicted Disease: {predicted_disease}")

    # Retrieve disease information
    symptoms, causes, treatment = get_disease_info(predicted_disease)

    # Display the results
    st.subheader(f"Predicted Disease: {predicted_disease}")
    st.write(f"**Symptoms:** {symptoms}")
    st.write(f"**Causes:** {causes}")
    st.write(f"**Treatment:** {treatment}")

