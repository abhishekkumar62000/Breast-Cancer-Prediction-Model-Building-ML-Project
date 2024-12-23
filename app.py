import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved KNN model
with open('KNN_7_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

# Set the background image and text color using CSS
def set_background(png_file):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{png_file}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    /* Set text color to black for all text elements */
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: white;
        text-align: center; /* Center align text */
    }}
    .stButton > button {{
        background-color: #4C4C6D; /* Button color */
        color: white; /* Text color for button */
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD; /* Button hover color */
        color: white;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to read and encode the image file
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Call the function with the background image
image_base64 = get_base64_image("image.png")  # Path to your image
set_background(image_base64)

# Title of the application
st.title("Breast Cancer Prediction")

st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 18px;">Input the following tumor characteristics and click <strong>Predict</strong> to determine whether the tumor is benign or malignant.</p>
    </div>
    """, unsafe_allow_html=True)

# Input fields for the features on the main screen
clump_thickness = st.slider('Clump Thickness', 1, 10, 1)
uniformity_cell_size = st.slider('Uniformity of Cell Size', 1, 10, 1)
uniformity_cell_shape = st.slider('Uniformity of Cell Shape', 1, 10, 1)
marginal_adhesion = st.slider('Marginal Adhesion', 1, 10, 1)
single_epithelial_cell_size = st.slider('Single Epithelial Cell Size', 1, 10, 1)
bare_nuclei = st.slider('Bare Nuclei', 1, 10, 1)
bland_chromatin = st.slider('Bland Chromatin', 1, 10, 1)
normal_nucleoli = st.slider('Normal Nucleoli', 1, 10, 1)
mitoses = st.slider('Mitoses', 1, 10, 1)

# Collect the user input into a feature vector
input_data = [[clump_thickness, uniformity_cell_size, uniformity_cell_shape, 
               marginal_adhesion, single_epithelial_cell_size, bare_nuclei, 
               bland_chromatin, normal_nucleoli, mitoses]]

# Feature scaling
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = knn_model.predict(input_data_scaled)

    # Display prediction result
    if prediction == 2:
        st.success("The tumor is predicted to be **Benign**.")
    else:
        st.error("The tumor is predicted to be **Malignant**.")

# Footer
st.markdown("---")
st.write("Created with ❤️ by Abhi")
