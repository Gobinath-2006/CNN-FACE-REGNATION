import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("face_recognition_augmented_3class.h5")

# Class labels (same order as training)
class_names = ['gobi', 'guru', 'sk']

st.title("CNN Face Recognition App")
st.write("Upload an image to predict the face")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")
    st.write(f"**Predicted Person:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
