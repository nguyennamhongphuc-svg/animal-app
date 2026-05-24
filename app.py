import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="AI Nhận diện Động vật", layout="wide")

# CSS để giao diện đẹp như ý bạn
st.markdown("""
    <style>
        .stApp {background-color: #FFF5F5;}
        .css-1r6slb0 {border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);}
    </style>
""", unsafe_allow_html=True)

# Load model với cache để tiết kiệm RAM
@st.cache_resource
def get_model():
    return tf.keras.models.load_model('animal_model.h5')

model = get_model()
classes = ['Mèo', 'Chó', 'Heo', 'Chuột', 'Voi', 'Chim', 'Cá', 'Ngựa', 'Thỏ', 'Rắn']

st.title("🐾 AI Nhận diện Động vật")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Vẽ con vật tại đây")
    canvas_result = st_canvas(
        fill_color="white", stroke_width=12,
        stroke_color="#FF8A65", background_color="white",
        height=300, width=300, drawing_mode="freedraw", key="canvas"
    )

with col2:
    st.subheader("Kết quả dự đoán")
    if canvas_result.image_data is not None:
        # Xử lý ảnh
        img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = 255 - img # Đảo màu
        img = img.reshape(1, 28, 28, 1) / 255.0
        
        # Dự đoán
        pred = model.predict(img)
        idx = np.argmax(pred)
        
        st.write(f"### Dự đoán: {classes[idx]}")
        st.progress(float(np.max(pred)))
