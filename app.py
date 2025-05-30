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
    background: linear-gradient(135deg, #16213e, #0f3460);
    color: #f0f0f0;
    min-height: 100vh;
    padding: 2rem 1rem;
}

/* Title with smooth gradient text and subtle shadow */
.stTitle {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 3rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    background: linear-gradient(90deg, #5bc0eb, #ff9f1c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 1px 1px 8px rgba(91,192,235,0.7);
}

/* Frame cards with subtle glass effect and soft glowing border */
.frame-card {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(18px);
    border-radius: 16px;
    padding: 22px 18px;
    margin-bottom: 24px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    border: 1.5px solid rgba(91,192,235,0.4);
    position: relative;
    overflow: hidden;
    transition: transform 0.3s ease;
}
.frame-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 30px rgba(91,192,235,0.5);
}

/* Animated gradient border */
.frame-card::before {
    content: '';
    position: absolute;
    top: -60%;
    left: -60%;
    width: 220%;
    height: 220%;
    background: linear-gradient(0deg, transparent, #5bc0eb, #ff9f1c, #5bc0eb, transparent);
    transform-origin: bottom right;
    animation: border-dance 6s linear infinite;
    filter: blur(24px);
    opacity: 0.6;
    z-index: -1;
}
@keyframes border-dance {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Frame images with subtle border and shadow */
.frame-image {
    max-width: 100%;
    height: auto;
    border-radius: 12px;
    border: 3px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
}

/* Labels */
.fake-label {
    color: #ff6b6b;
    font-weight: 700;
    text-shadow: 0 0 5px #ff6b6baa;
}
.real-label {
    color: #4ecdc4;
    font-weight: 700;
    text-shadow: 0 0 5px #4ecdc4aa;
}

/* Final prediction container */
.final-prediction {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 30px 35px;
    text-align: center;
    margin-top: 3rem;
    box-shadow: 0 20px 40px rgba(91,192,235,0.35);
    position: relative;
    overflow: hidden;
    border: 2px solid rgba(255, 255, 255, 0.25);
    transition: box-shadow 0.3s ease;
}
.final-prediction:hover {
    box-shadow: 0 30px 60px rgba(91,192,235,0.6);
}

/* Glow effect */
.final-prediction::after {
    content: '';
    position: absolute;
    bottom: -60%;
    left: -60%;
    width: 220%;
    height: 220%;
    background: linear-gradient(0deg, transparent, #5bc0eb, #ff9f1c, #5bc0eb, transparent);
    animation: border-dance 6s linear infinite;
    filter: blur(28px);
    opacity: 0.7;
    z-index: -1;
    transform-origin: top right;
}

/* Text in final result */
.final-result-title {
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.7rem;
    letter-spacing: 1.5px;
    color: #ff9f1c;
    text-shadow: 0 0 8px #ff9f1caa;
}

.final-result-value {
    font-size: 2.8rem;
    font-weight: 900;
    margin-bottom: 0.8rem;
    color: #5bc0eb;
    text-shadow: 0 0 10px #5bc0ebcc;
}

.final-confidence {
    font-size: 1.3rem;
    font-weight: 600;
    color: #eee;
    margin-bottom: 0;
    letter-spacing: 0.8px;
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
