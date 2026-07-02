import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown 
import os
# Page configuration
st.set_page_config(page_title="ECG Classifier")


@st.cache_resource
def load_model():
    file_id = "1h5EMZTaGM6KnaqrS1Ur6TXUKzdZPYZIm"
    url = f"https://drive.google.com/uc?id={file_id}"

    output = "ecg_mlp_model.keras"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    model = tf.keras.models.load_model(output)
    return model

model = load_model()



# Class names in the exact training order
class_names = [
    "Myocardial Infarction",
    "History of MI",
    "Abnormal Heartbeat",
    "Normal"
]

# Image size matching the model training
img_size = (128, 128)

# Title and description
st.title("ECG Image Classifier")
st.write("Upload an ECG image to classify it into one of 4 conditions using the MLP model.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an ECG image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Image preprocessing matching the training pipeline
    image = image.convert("RGB")
    image = image.resize(img_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction button
    if st.button("Classify Image"):

        prediction = model.predict(image_array)
        probabilities = prediction[0]

        predicted_index = np.argmax(probabilities)
        predicted_class = class_names[predicted_index]
        confidence = probabilities[predicted_index] * 100

        st.success(f"Classification: **{predicted_class}** (Confidence: {confidence:.2f}%)")

        st.subheader("Confidence level for each class:")

        for i in range(len(class_names)):
            class_name = class_names[i]
            class_probability = probabilities[i] * 100
            st.write(f"{class_name}: {class_probability:.2f}%")
            st.progress(int(class_probability))