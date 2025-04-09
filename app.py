import streamlit as st
import whisper
import tempfile
import os
from moviepy.editor import VideoFileClip

st.set_page_config(page_title="Transcripteur de conférences", layout="centered")

st.title("🎤 Transcripteur vidéo → texte pour conférences")

uploaded_file = st.file_uploader("📤 Téléverse ta vidéo de conférence", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with st.spinner("📽️ Traitement de la vidéo..."):
        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_file.read())
            video_path = tmp_video.name

        # Extraction de l'audio
        st.info("🔊 Extraction de l'audio...")
        audio_path = video_path.replace(".mp4", ".mp3")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path)

        # Chargement du modèle Whisper
        st.info("🧠 Chargement du modèle Whisper...")
        model = whisper.load_model("base")  # tu peux tester "small" ou "medium" selon ta machine

        # Transcription
        st.info("✍️ Transcription en cours...")
        result = model.transcribe(audio_path)

        # Affichage
        st.success("✅ Transcription terminée !")
        st.subheader("📝 Texte extrait :")
        st.text_area("Résultat", result["text"], height=300)

        # Téléchargement
        st.download_button("⬇️ Télécharger le texte", result["text"], file_name="transcription.txt")

        # Nettoyage des fichiers temporaires
        os.remove(video_path)
        os.remove(audio_path)
