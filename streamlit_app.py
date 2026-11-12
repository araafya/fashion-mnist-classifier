import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Fashion-MNIST Classifier",
    layout="centered"
)

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(img: Image.Image, model):
    img = img.convert("L").resize((28, 28))
    arr = np.array(img, dtype="float32")

    # Data Fashion-MNIST berupa objek terang di latar gelap.
    # Jika latar gambar yang di-upload terang (rata-rata piksel tepi > 127), warnanya dibalik.
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:
        arr = 255.0 - arr

    arr = arr / 255.0  # normalisasi 0-1, sama seperti di notebook

    # Bentuk input mengikuti model: (1, 28, 28, 1) untuk CNN atau (1, 784) untuk ANN
    model_input = arr.reshape((1,) + tuple(model.input_shape[1:]))
    return model_input, arr


def main():
    st.title("Fashion-MNIST Classifier")
    st.write("Upload gambar pakaian, lalu model akan memprediksi jenis pakaiannya.")
    st.caption("Kelas: " + ", ".join(CLASS_NAMES))

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang di-upload", width=250)

        if st.button("Predict"):
            model = load_model()
            processed_image, preview = preprocess_image(image, model)

            with st.spinner("Predicting..."):
                prediction = model.predict(processed_image)[0]
                pred_class = int(np.argmax(prediction))
                confidence = float(prediction[pred_class]) * 100

            st.success(f"Prediction: {CLASS_NAMES[pred_class]}")
            st.info(f"Confidence: {confidence:.2f}%")

            st.image(preview, caption="Input model (28×28 grayscale)", width=150, clamp=True)

            st.write("Class Probabilities:")
            for i, prob in enumerate(prediction):
                st.progress(float(prob))
                st.write(f"{CLASS_NAMES[i]}: {float(prob) * 100:.2f}%")


if __name__ == "__main__":
    main()
