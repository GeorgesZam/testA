import streamlit as st
import whisper
import tempfile
import os
import ffmpeg

st.set_page_config(page_title="Transcripteur de conférences", layout="centered")
st.title("🎤 Transcripteur vidéo → texte pour conférences")

uploaded_file = st.file_uploader("📤 Téléverse ta vidéo de conférence", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with st.spinner("📽️ Traitement de la vidéo..."):
        # Sauvegarde temporaire de la vidéo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_file.read())
            video_path = tmp_video.name

        # Conversion en audio avec ffmpeg
        st.info("🔊 Extraction de l'audio avec ffmpeg...")
        audio_path = video_path.replace(".mp4", ".wav")

        ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run(quiet=True, overwrite_output=True)

        # Transcription avec Whisper
        st.info("🧠 Transcription avec Whisper...")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)

        # Résultat
        st.success("✅ Transcription terminée !")
        st.subheader("📝 Texte extrait :")
        st.text_area("Résultat", result["text"], height=300)

        st.download_button("⬇️ Télécharger la transcription", result["text"], file_name="transcription.txt")

        # Nettoyage
        os.remove(video_path)
        os.remove(audio_path)
