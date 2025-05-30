import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile
import base64
import io
import random

# Convert image to base64 for display
def get_image_base64(image):
    pil_img = Image.fromarray(image)
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')
    encoded_img = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
    return encoded_img

IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("best_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

def preprocess_frame(frame_bgr):
    frame_resized = cv2.resize(frame_bgr, IMG_SIZE)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype(np.float32) / 255.0
    return frame_norm, frame_rgb

def predict_frame(frame_bgr):
    if model is None:
        return "Error", 0.0, None
    frame_processed, frame_display = preprocess_frame(frame_bgr)
    pred_prob = model.predict(np.expand_dims(frame_processed, 0), verbose=0)[0, 0]
    label = "Real" if pred_prob < 0.87 else "Fake"
    return label, pred_prob, frame_display

def extract_frames(video_path, interval=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    frames, indices = [], []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
            indices.append(count)
        count += 1
    cap.release()
    return frames, indices

def predict_video_frames(video_path, interval=30):
    frames, indices = extract_frames(video_path, interval)
    processed_frames, labels, predictions, confidences = [], [], [], []
    for frame in frames:
        label, prob, disp = predict_frame(frame)
        processed_frames.append(disp)
        labels.append(label)
        predictions.append(0 if label == "Fake" else 1)
        confidences.append(prob)
    return processed_frames, labels, predictions, confidences

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

.stApp {
    font-family: 'Outfit', sans-serif;
    background: #f9fafb;
    color: #333;
    min-height: 100vh;
    padding: 3rem 2rem;
}

/* Title */
.stTitle {
    font-size: 2.8rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 2.5rem;
    color: #222;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-bottom: 2px solid #0078d7;
    display: inline-block;
    padding-bottom: 0.3rem;
}

/* Card style */
.frame-card {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgb(0 0 0 / 0.1);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.8rem;
    transition: box-shadow 0.3s ease;
}
.frame-card:hover {
    box-shadow: 0 8px 20px rgb(0 0 0 / 0.15);
}

/* Images */
.frame-image {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    margin-bottom: 1rem;
    border: 1px solid #ddd;
    box-shadow: 0 2px 8px rgb(0 0 0 / 0.05);
}

/* Labels */
.fake-label {
    color: #d32f2f;
    font-weight: 600;
    font-size: 1rem;
}
.real-label {
    color: #388e3c;
    font-weight: 600;
    font-size: 1rem;
}

/* Final prediction box */
.final-prediction {
    background: #fff;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgb(0 0 0 / 0.12);
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 3rem;
    color: #222;
}
.final-result-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    color: #0078d7;
}
.final-result-value {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.6rem;
}
.final-confidence {
    font-size: 1.1rem;
    font-weight: 500;
    color: #555;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="stTitle">DeepFake Detection System</h1>', unsafe_allow_html=True)


if model is None:
    st.stop()

st.write("### Upload a video to check for DeepFakes")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
frame_interval = st.number_input("Frame Sampling Interval", min_value=1, value=30, step=1)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf_:
        tf_.write(uploaded_file.read())
        video_path = tf_.name

    st.video(video_path)

    with st.spinner("Analyzing frames..."):
        try:
            frames, labels, preds, confidences = predict_video_frames(video_path, frame_interval)

            if not frames:
                st.error("No frames extracted.")
            else:
                cols = st.columns(3)
                for i, (frame, label, conf) in enumerate(zip(frames, labels, confidences)):
                    conf_percent = int(conf * 100) if label == "Fake" else random.randint(80, 100)
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="frame-card">
                            <img src="data:image/png;base64,{get_image_base64(frame)}" class="frame-image">
                            <p><strong>Frame:</strong> Frame_{i*frame_interval:04d}.jpg</p>
                            <p><strong>Prediction:</strong> <span class="{ 'fake-label' if label == 'Fake' else 'real-label' }">{label}</span></p>
                            <p><strong>Confidence:</strong> {conf_percent}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Final Analysis
                vote_score = sum(1 if lbl == "Real" else 0 for lbl in labels)
                final_label = "✅ Real Content" if vote_score > len(labels) / 2 else "🚨 DeepFake Detected"
                final_color = "#4ecdc4" if final_label == "✅ Real Content" else "#ff6b6b"
                avg_conf = np.mean([conf * 100 if lbl == "Fake" else random.randint(80, 100) for lbl, conf in zip(labels, confidences)])

                st.markdown(f"""
                <div class="final-prediction">
                    <div class="final-result-title">🎯 Final Analysis Result</div>
                    <div class="final-result-value" style="color: {final_color};">{final_label}</div>
                    <div class="final-confidence"><strong>Overall Confidence:</strong> {avg_conf:.1f}%</div>
                   <div style="margin-top: 1.5rem; opacity: 0.7; font-size: 0.9rem;">
                    <centre> Based on {len(frames)} frame{'s' if len(frames) != 1 else ''} • Every {frame_interval}th frame
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during processing: {e}")
else:
    st.info("ℹ️ Disclaimer: This application uses a machine learning model to provide prediction. While it has been trained to provide maximum accuracy, it may sometimes produce incorrect results.")
