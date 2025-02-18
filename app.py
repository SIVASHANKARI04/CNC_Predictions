import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import os

import base64
def set_image_local(image_path):
    with open(image_path, "rb") as file:
        img = file.read()
    base64_image = base64.b64encode(img).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
            background-position: fit;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_image_local(r"pic2.jpg")
# Function to load the Keras model
def load_model_file():
    custom_objects = {"LeakyReLU": LeakyReLU}
    try:
        model_path = r"model.h5"
        if os.path.exists(model_path):
            return load_model(model_path, custom_objects=custom_objects)
        else:
            st.error("Model file not found. Check the path!")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load the scaler
def load_scaler_file():
    try:
        scaler_path = "scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                return pickle.load(f)
        else:
            st.error("Scaler file not found. Check the path!")
            return None
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

# Load the model and scaler
model = load_model_file()
scaler = load_scaler_file()

# Streamlit UI
st.title("CNC Machine Failure Prediction")

# User inputs
Type_select = st.selectbox('Select Job Type', ['Low', 'Medium', 'High'])
Type_map = {'Low': 1, 'High': 0, 'Medium': 2}
Type = Type_map.get(Type_select)

Air_Temperature = st.number_input('Enter Air Temperature', value=0.0, step=0.01, min_value=0.0)
Process_temperature = st.number_input('Enter Process temperature', value=0.0, step=0.01, min_value=0.0)
Rotational_speed_1 = st.number_input('Enter Rotational speed', min_value=0.1)
Rotational_speed = 1 / (Rotational_speed_1 ** 2)
Torque = st.number_input('Enter Torque', value=0.0, step=0.01, min_value=0.0)
Tool_wear = st.number_input('Enter Tool wear', value=0.0, step=0.01, min_value=0.0)

if st.button('Predict'):
    if model and scaler:
        # Prepare input data
        c1 = np.array([[Type, Air_Temperature, Process_temperature, Rotational_speed, Torque, Tool_wear]])

        try:
            # Transform input data using the scaler
            c2 = scaler.transform(c1)

            # Make prediction
            predicted_output = model.predict(c2)
            predicted_class = np.argmax(predicted_output, axis=1)

            # Display prediction
            st.write(f"Predicted Class: {predicted_class[0]}")

            # Class mapping
            class_map = {
                0: "Heat Dissipation Failure",
                1: "No Failure",
                2: "Overstrain Failure",
                3: "Power Failure",
                4: "Random Failures",
                5: "Tool Wear Failure"
            }
            predicted_status = class_map.get(predicted_class[0], "Unpredictable input. Re-enter correct input")
            st.write(f"Predicted Status: {predicted_status}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model or scaler is not loaded.")
