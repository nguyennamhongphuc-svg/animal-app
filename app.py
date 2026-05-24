import streamlit as st
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from PIL import Image

model=load_model("animal_sketch_model.h5")

classes=[
    "cat",
    "dog",
    "fish",
    "bird",
    "rabbit",
    "lion",
    "tiger",
    "elephant",
    "monkey",
    "horse"
]

st.set_page_config(
    page_title="AI Drawing Recognition",
    layout="centered"
)

st.title("AI Drawing Recognition System")

st.write("Upload a drawing and AI will predict the animal.")

uploaded_file=st.file_uploader(
    "Upload Drawing",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image=Image.open(uploaded_file)

    st.image(image,width=300)

    img=np.array(image)

    img=cv2.resize(img,(28,28))

    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    img=255-img

    img=img.astype("float32")/255.0

    img=img.reshape(1,28,28,1)

    prediction=model.predict(img)[0]

    index=np.argmax(prediction)

    animal=classes[index]

    confidence=float(np.max(prediction)*100)

    st.success(f"Prediction: {animal}")

    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("All Predictions")

    for i,c in enumerate(classes):
        st.write(f"{c}: {prediction[i]*100:.2f}%")
