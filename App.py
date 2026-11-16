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

disease_info = {
    'Cassava Bacterial Blight': {
        'cause': 'Caused by bacteria (Xanthomonas). It spreads through infected cuttings, rain splash and dirty farm tools.',
        'solution': '1. Remove and burn all infected leaves immediately.\n2. Use clean, disease-free cuttings for planting.\n3. Disinfect cutlass/hoe with bleach water after use.\n4. Avoid working in the farm when leaves are wet.\n5. Plant resistant varieties if available.',
        'prevention': 'Use certified planting material and practice crop rotation. Do not plant cassava on the same land every year.'
    },
    'Cassava Brown Streak': {
        'cause': 'Caused by virus spread by whiteflies. It damages the roots from inside.',
        'solution': '1. Uproot and destroy infected plants - do not leave them in farm.\n2. Use virus-free, certified cuttings only.\n3. Control whiteflies with neem oil or approved insecticide.\n4. Harvest early if infection is severe to save some roots.',
        'prevention': 'Always plant clean cuttings and check farm weekly for whiteflies.'
    },
    'Cassava Mosaic': {
        'cause': 'Caused by virus spread by whiteflies and infected cuttings. This is the most common cassava disease.',
        'solution': '1. Uproot and burn infected plants immediately.\n2. Do NOT use cuttings from infected plants.\n3. Plant resistant varieties like TMS varieties.\n4. Spray neem oil to control whiteflies.\n5. Plant early at start of rainy season.',
        'prevention': 'Use only resistant, certified cuttings. Check your farm every week.'
    },
    'Healthy Cassava': {
        'cause': 'Your cassava leaf looks healthy. No disease detected.',
        'solution': 'No treatment needed. Your plant is doing well.',
        'prevention': 'Continue good farm practice: weed regularly, use clean tools, and monitor leaves weekly.'
    },
    'Healthy Plantain': {
        'cause': 'Your plantain leaf looks healthy. No disease detected.',
        'solution': 'No treatment needed. Your plant is doing well.',
        'prevention': 'Continue good farm practice: clear weeds around plant, apply manure, and monitor leaves weekly.'
    },
    'Plantain Black Sigatoka': {
        'cause': 'Caused by fungus. It causes black spots on leaves, leaves dry up and bunch becomes small.',
        'solution': '1. Cut off and burn badly infected leaves.\n2. Improve air flow - do not plant too close.\n3. Apply fungicide like Mancozeb or Copper-based fungicide as directed.\n4. Apply organic manure / fertilizer to make plant strong.\n5. Remove weeds around the plant.',
        'prevention': 'Space plants well, remove old leaves regularly, and plant resistant varieties.'
    }
}

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
        predicted_class = class_names[idx]

        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: {conf:.2f}%")
        st.bar_chart(pred[0])
        info = disease_info.get(predicted_class)
        if info:
            st.markdown("---")
            st.subheader(f"🩺 Diagnosis: {predicted_class}")
            st.write(f"**Wetin cause am:** {info['cause']}")
            st.subheader("✅ Wetin you go do (Solution):")
            st.write(info['solution'])
            st.subheader("🛡️ How to prevent am next time:")
            st.write(info['prevention'])

st.markdown("---")
st.write("Built with Streamlit | Model: model.h5")
