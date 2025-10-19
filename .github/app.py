import streamlit as st
from fer import FER
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import os
import re

# ---------- Config ----------
st.set_page_config(page_title="FeelTune AI", layout="centered")

# ---------- Ambient Background ----------
st.markdown("""
<style>
@keyframes gradientBG {
    0%{background-color:#1a2a6c;}
    25%{background-color:#b21f1f;}
    50%{background-color:#fdbb2d;}
    75%{background-color:#00c6ff;}
    100%{background-color:#1a2a6c;}
}
body {
    animation: gradientBG 30s ease infinite;
    color: #eaeaea;
    font-family: 'Arial', sans-serif;
}
h1, h2, h3 { color: #ffffff; }
.stButton>button {
    background-color: rgba(255,255,255,0.2);
    color: #ffffff;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
}
.mood-text {
    font-size: 1.4em;
    font-weight: bold;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 15px;
    text-align:center;
}
.mood-happy { background: linear-gradient(90deg,#fddb92,#d1fdff); color:#1a2a6c; }
.mood-sad { background: linear-gradient(90deg,#667db6,#0082c8); color:#f5f5f5; }
.mood-angry { background: linear-gradient(90deg,#ff416c,#ff4b2b); color:#fff; }
.mood-surprise { background: linear-gradient(90deg,#f9ff00,#ff4b1f); color:#1a1a2e; }
.mood-fear { background: linear-gradient(90deg,#0f0c29,#302b63,#24243e); color:#fff; }
.mood-disgust { background: linear-gradient(90deg,#11998e,#38ef7d); color:#1a1a2e; }
</style>
""", unsafe_allow_html=True)

st.title("🎧 FeelTune AI — Mood-Based Music Player")

# ---------- Mappings ----------
EMOTION_TO_YOUTUBE = {
    "happy": "https://www.youtube.com/watch?v=6LjLdctwN7k&list=PLkfB18MbHxMTFqEnsVXjzLJOD7gl7E265",
    "sad":   "https://www.youtube.com/watch?v=AKUk1v3rBvc&list=PLkfB18MbHxMQgz5uDNRTO79c0-3GKINS8",
    "angry": "https://www.youtube.com/watch?v=jFGKJBPFdUA&list=PLxNm0dqHxmlupV3dr7uq4Rl8L5nwlGKQA",
    "surprise": "https://youtu.be/uVM5G2rfy14?si=q9FuH_R-NcIvt9sP",
    "fear":  "https://youtu.be/8afBXZawfQw?si=vwYfF7cBelS0thtL",
    "disgust":"https://www.youtube.com/watch?v=RWts_-gDZDY&list=PLkfB18MbHxMQ5nu-u_85_O0sYXuRYqE9p"
}

MOOD_MESSAGES = {
    "happy": "You're feeling great! Enjoy this upbeat music 🎵",
    "sad": "Playing music to heal your sad mood 💙",
    "angry": "Release your anger with some punchy tunes! 🔥",
    "surprise": "Surprise! Let's energize your day with music 🎉",
    "fear": "Don't worry, we will overcome your fear together 🌌",
    "disgust": "Calm your senses with mellow tunes 🌿"
}

HISTORY_CSV = "mood_history.csv"
detector = FER(mtcnn=True)

# ---------- UI ----------
st.markdown("Capture your face and FeelTune AI will instantly play mood-matching music! 🎶")

img_file = st.camera_input("📷 Take a photo")
col1, col2 = st.columns([2,1])

with col1:
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Captured Image", use_column_width=True)
        img_np = np.array(img)

        with st.spinner("Detecting emotion..."):
            try:
                faces = detector.detect_emotions(img_np)
            except Exception as e:
                st.error(f"Error detecting emotions: {e}")
                faces = []

            if faces:
                best = max(faces, key=lambda x: sum(x["emotions"].values()))
                emotions = best["emotions"]
                emotions.pop("neutral", None)

                if not emotions:
                    st.info("No significant emotion detected.")
                else:
                    top_emotion, top_score = max(emotions.items(), key=lambda x: x[1])
                    st.success(f"Detected emotion: **{top_emotion.capitalize()}** (confidence: {top_score:.2f})")

                    # Mood message
                    message = MOOD_MESSAGES.get(top_emotion, "")
                    st.markdown(f'<div class="mood-text mood-{top_emotion}">{message}</div>', unsafe_allow_html=True)

                    # Emotion probabilities
                    df_em = pd.DataFrame(sorted(emotions.items(), key=lambda x: x[1], reverse=True),
                                         columns=["Emotion", "Confidence"])
                    st.table(df_em.set_index("Emotion"))

                    # Auto-play YouTube
                    yt_url = EMOTION_TO_YOUTUBE.get(top_emotion)
                    if yt_url:
                        video_id_search = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", yt_url)
                        if video_id_search:
                            video_id = video_id_search.group(1)
                            st.markdown(f"""
                            <iframe width="100%" height="315"
                            src="https://www.youtube.com/embed/{video_id}?autoplay=1&mute=0"
                            title="YouTube video player" frameborder="0"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowfullscreen>
                            </iframe>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"[Open Playlist]({yt_url})")

                    # Save mood history
                    row = {
                        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
                        "emotion": top_emotion,
                        "score": float(top_score)
                    }
                    if os.path.exists(HISTORY_CSV):
                        hist_df = pd.read_csv(HISTORY_CSV)
                        hist_df = pd.concat([hist_df, pd.DataFrame([row])], ignore_index=True)
                    else:
                        hist_df = pd.DataFrame([row])
                    hist_df.to_csv(HISTORY_CSV, index=False)
            else:
                st.warning("No face detected. Ensure proper lighting and position.")

with col2:
    st.markdown("### 📈 Mood History")
    if os.path.exists(HISTORY_CSV):
        df_hist = pd.read_csv(HISTORY_CSV)
        st.dataframe(df_hist.tail(10))
        st.markdown("**Counts by emotion:**")
        st.bar_chart(df_hist["emotion"].value_counts())
    else:
        st.info("Mood history will appear here after detecting your first emotion.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("<center>DEVELOPED BY AETHERION TEAM</center>", unsafe_allow_html=True)
