import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('animal_model.h5')

model = load_model()
classes = ['Mèo', 'Chó', 'Heo', 'Chuột', 'Voi', 'Chim', 'Cá', 'Ngựa', 'Thỏ', 'Rắn']

st.title("🐾 AI Nhận dạng Động vật")
st.write("Hãy vẽ con vật bạn thích vào khung dưới đây:")

# Canvas vẽ
canvas_result = st_canvas(
    fill_color="white", stroke_width=10,
    stroke_color="black", background_color="white",
    height=280, width=280, drawing_mode="freedraw", key="canvas"
)

if canvas_result.image_data is not None:
    # Tiền xử lý ảnh
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28))
    # Đảo ngược màu vì model train trên nền đen (nét trắng)
    img = 255 - img 
    img = img.reshape(1, 28, 28, 1) / 255.0
    
    # Dự đoán
    pred = model.predict(img)
    idx = np.argmax(pred)
    
    st.subheader(f"AI đoán đây là: {classes[idx]}")
    st.progress(float(np.max(pred)))
