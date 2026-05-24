import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np
import os

# Cấu hình giao diện
st.set_page_config(page_title="AI Nhận diện Động vật", layout="wide")

# CSS tạo phong cách hiện đại
st.markdown("""
    <style>
        .stApp {background-color: #FFF5F5;}
        .result-box {background: white; padding: 20px; border-radius: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Load model với xử lý lỗi
@st.cache_resource
def load_model():
    if os.path.exists('animal_model.h5'):
        return tf.keras.models.load_model('animal_model.h5')
    return None

model = load_model()
classes = ['Mèo', 'Chó', 'Heo', 'Chuột', 'Voi', 'Chim', 'Cá', 'Ngựa', 'Thỏ', 'Rắn']

st.title("🐾 AI Nhận diện Động vật")

if model is None:
    st.error("Chưa tìm thấy file 'animal_model.h5'. Hãy chắc chắn bạn đã upload nó lên GitHub!")
else:
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
            # Xử lý ảnh: lấy kênh alpha, chuyển sang đen trắng
            img = canvas_result.image_data[:, :, 0:3]
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (28, 28))
            img = 255 - img  # Đảo màu để nét vẽ thành màu trắng trên nền đen
            img = img.reshape(1, 28, 28, 1) / 255.0
            
            # Dự đoán
            pred = model.predict(img)
            idx = np.argmax(pred)
            
            st.markdown(f'<div class="result-box"><h3>Dự đoán: {classes[idx]}</h3>', unsafe_allow_html=True)
            st.progress(float(np.max(pred)))
            st.write(f"Độ tin cậy: {np.max(pred)*100:.2f}%</div>", unsafe_allow_html=True)
