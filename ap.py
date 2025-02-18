import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import os
import base64

# Set background image from local file
def set_image_local(image_path):
    try:
        with open(image_path, "rb") as file:
            img = file.read()
        base64_image = base64.b64encode(img).decode("utf-8")
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{base64_image}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error loading background image: {e}")

# Set the background image
set_image_local("Predictive-Maintenance-with-Machine-Learning_-A-Complete-Guide.webp")

# Function to load the Keras model
def load_model_file():
    custom_objects = {"LeakyReLU": LeakyReLU}
    model_path = "model1.h5"
    if os.path.exists(model_path):
        try:
            return load_model(model_path, custom_objects=custom_objects)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error("Model file not found. Check the path!")
        return None

# Function to load the scaler pickle file
def load_scaler_file():
    scaler_path = "CNC_PM.pkl"
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            return None
    else:
        st.error("Scaler file not found. Check the path!")
        return None

# Load model and scaler
model = load_model_file()
scaler = load_scaler_file()

# Streamlit UI
st.title("CNC Machine Failure Prediction")
st.write("Provide the required features to predict machine failure.")

# --- Input Features ---

# 1. Job Type
Type_select = st.selectbox('Select Job Type', ['Low', 'Medium', 'High'])
Type_map = {'Low': 1, 'Medium': 2, 'High': 0}
job_type = Type_map.get(Type_select)

# 2. Air Temperature
Air_Temperature = st.number_input('Enter Air Temperature', value=0.0, step=0.01, min_value=0.0)

# 3. Process Temperature
Process_temperature = st.number_input('Enter Process Temperature', value=0.0, step=0.01, min_value=0.0)

# 4. Rotational Speed
# Here, we take a parameter and then transform it using 1/(x^2)
Rotational_speed_input = st.number_input('Enter Rotational Speed parameter (for transformation)', min_value=0.1)
Rotational_speed = 1 / (Rotational_speed_input ** 2)

# 5. Torque
Torque = st.number_input('Enter Torque', value=0.0, step=0.01, min_value=0.0)

# 6. Tool Wear
Tool_wear = st.number_input('Enter Tool Wear', value=0.0, step=0.01, min_value=0.0)

# --- Prediction ---
if st.button('Predict'):
    if model is not None and scaler is not None:
        # Create input array with features in the correct order
        input_features = np.array([[job_type, Air_Temperature, Process_temperature, Rotational_speed, Torque, Tool_wear]])
        try:
            # Scale the features using the loaded scaler
            scaled_features = scaler.transform(input_features)
            
            # Get model prediction
            predicted_output = model.predict(scaled_features)
            predicted_class = np.argmax(predicted_output, axis=1)[0]
            
            # Map predicted class to status (adjust as per your model's training)
            class_map = {
                0: "Heat Dissipation Failure",
                1: "No Failure",
                2: "Overstrain Failure",
                3: "Power Failure",
                4: "Random Failures",
                5: "Tool Wear Failure"
            }
            predicted_status = class_map.get(predicted_class, "Unpredictable input. Re-enter correct input")
            
            st.success(f"Predicted Class: {predicted_class}")
            st.success(f"Predicted Status: {predicted_status}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model or scaler is not loaded.")
