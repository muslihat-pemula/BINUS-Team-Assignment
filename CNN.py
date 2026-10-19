import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10 Image Classification",
    page_icon="💀",
    layout="wide"
)

MODEL_CANDIDATES = [
    "cifar10_cnn_model.keras",
    "model_final.keras",
    "model_cifar10_kelompok5.keras"
]

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CLASS_LABELS_ID = {
    'airplane': 'Pesawat Terbang',
    'automobile': 'Mobil',
    'bird': 'Burung',
    'cat': 'Kucing',
    'deer': 'Rusa',
    'dog': 'Anjing',
    'frog': 'Katak',
    'horse': 'Kuda',
    'ship': 'Kapal Laut',
    'truck': 'Truk'
}

IMG_SIZE = (32, 32)

@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return tf.keras.models.load_model(path), path
    return None, None

def preprocess(img: Image.Image) -> np.ndarray:
    resized_img = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.asarray(resized_img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

with st.sidebar:
    st.header("About the Model")
    st.markdown("""
    - **Architecture**: CNN (Conv2D + Dropout + Dense)
    - **Dataset**: CIFAR-10
    - **Input size**: 32×32 pixels RGB
    - **Test accuracy**: High Performance CNN
    """)
    st.divider()
    st.header("Known classes")
    for idx, name in enumerate(CLASS_NAMES):
        st.write(f"**{idx}.** {name} (*{CLASS_LABELS_ID[name]}*)")
    st.divider()
    st.header("☢IMPORTANT☢")
    st.warning("""
    - Make sure the picture had only 1 main object.
    - The object mostly need to be centered and fill the frame, otherwise? try it by ur self.
    - **Objects that is clearly not in the "10 Known Classes" will be forced into one of the classes.**
    - And remember.. No Russians.
    """)

st.title("CIFAR-10 Image Classification")

model, loaded_path = load_model()

if model is None:
    st.error("Model file (.keras) not found in directory. Please place 'cifar10_cnn_model.keras' in the same folder.")
    st.stop()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.write("Upload gambar untuk diprediksi oleh model CNN.")
    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.caption("Gambar Asli")
            st.image(img, width=120)
        with img_col2:
            st.caption("Input Model (32x32)")
            resized_preview = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
            st.image(resized_preview, width=120)
            
        input_tensor = preprocess(img)
        probs = model.predict(input_tensor, verbose=0)[0]
        
        top_idx = int(np.argmax(probs))
        top_class = CLASS_NAMES[top_idx]
        top_class_id = CLASS_LABELS_ID[top_class]
        confidence = float(probs[top_idx]) * 100
        
        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Prediksi Utama", f"{top_class}", delta=top_class_id)
        with res_col2:
            st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
            
        if confidence < 50.0:
            st.warning("⚠️ Tingkat keyakinan rendah (<50%). Gambar mungkin berada di luar 10 kelas CIFAR-10.")

with right_col:
    if uploaded_file is not None:
        st.subheader("Probabilitas Tiap Kelas")
        
        sorted_indices = np.argsort(probs)[::-1]
        sorted_classes = [CLASS_NAMES[i] for i in sorted_indices]
        sorted_probs = [float(probs[i]) for i in sorted_indices]
        
        chart_data = pd.DataFrame({
            "Class": sorted_classes,
            "Probability": sorted_probs
        }).set_index("Class")
        
        st.bar_chart(chart_data, horizontal=True, height=280)
        
        with st.expander("Lihat Tabel Probabilitas Detail"):
            df_table = pd.DataFrame({
                "Kelas": [f"{CLASS_NAMES[i]} ({CLASS_LABELS_ID[CLASS_NAMES[i]]})" for i in sorted_indices],
                "Probabilitas (%)": [f"{probs[i]*100:.2f}%" for i in sorted_indices],
                "Skor Mentah": [f"{probs[i]:.6f}" for i in sorted_indices]
            })
            st.dataframe(df_table, use_container_width=True, hide_index=True)
    else:
        st.info("Silakan unggah gambar di panel sebelah kiri untuk menganalisis hasil prediksi.")