import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cassava & Plantain Disease Detector", page_icon="🌿")
st.title("🌿 Cassava & Plantain Disease Detector")
st.write("Upload a leaf image to detect disease")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

class_names = ['Cassava Bacterial Blight', 'Cassava Brown Streak', 'Cassava Mosaic', 'Healthy Cassava', 'Healthy Plantain', 'Plantain Black Sigatoka']

uploaded = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        idx = np.argmax(pred)
        conf = np.max(pred) * 100
        st.success(f"Prediction: **{class_names[idx]}**")
        st.info(f"Confidence: {conf:.2f}%")
        st.bar_chart(pred[0])

st.markdown("---")
st.write("Built with Streamlit | Model: model.h5")
