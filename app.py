import os
import re
import tempfile
from datetime import timedelta

import streamlit as st
from pytube import YouTube
from openai import OpenAI

# Initialize OpenAI client (requires OPENAI_API_KEY env var)
client = OpenAI()

# ---------- Utility Functions ----------

def is_valid_youtube_url(url: str) -> bool:
    pattern = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+"
    return bool(re.match(pattern, url.strip()))

def format_duration(seconds: int) -> str:
    return str(timedelta(seconds=seconds))

def select_smallest_audio_stream(yt: YouTube):
    streams = yt.streams.filter(only_audio=True)
    if not streams:
        return None
    # Pick the stream with the smallest approximate filesize
    smallest = min(streams, key=lambda s: getattr(s, "filesize_approx", float("inf")))
    return smallest

def download_audio_from_youtube(url: str) -> tuple[str, dict]:
    yt = YouTube(url)
    stream = select_smallest_audio_stream(yt)
    if not stream:
        raise RuntimeError("No audio-only stream available for this video.")
    meta = {
        "title": yt.title,
        "author": yt.author,
        "length": yt.length,
        "views": yt.views,
        "publish_date": yt.publish_date.isoformat() if yt.publish_date else None,
        "mime_type": stream.mime_type,
        "abr": getattr(stream, "abr", None),
    }

    temp_dir = tempfile.mkdtemp(prefix="yt_audio_")
    filename_base = re.sub(r"[^\w\-_. ]", "_", yt.title) or "audio"
    # Let pytube decide correct extension based on mime_type
    filepath = stream.download(output_path=temp_dir, filename=filename_base)
    return filepath, meta

def transcribe_audio(filepath: str, language: str | None = None) -> str:
    with open(filepath, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language if language and language != "auto" else None
        )
    return transcript.text

# ---------- Streamlit App ----------

st.set_page_config(page_title="YouTube -> Text Transcriber", page_icon="🎧", layout="centered")
st.title("YouTube Audio Transcriber 🎧")
st.caption("Paste a YouTube link, extract its audio, and convert it to text using OpenAI Whisper.")

with st.expander("Setup Notes", expanded=False):
    st.write(
        "Ensure your environment has:\n"
        "- Environment variable OPENAI_API_KEY set with your OpenAI API key.\n"
        "- Packages installed: streamlit, pytube, openai (>=1.0.0).\n"
        "This app uses OpenAI Whisper (model: whisper-1) for transcription."
    )

url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
lang = st.selectbox(
    "Transcription language (optional, improves accuracy)",
    options=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "hi", "ja", "ko", "zh"],
    index=0,
    help="Choose 'auto' to let the model detect language automatically."
)
max_size_mb = st.number_input(
    "Max audio file size to process (MB)",
    min_value=1,
    max_value=50,
    value=24,
    help="To avoid API limits, videos producing audio larger than this will be skipped."
)

if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "meta" not in st.session_state:
    st.session_state.meta = None

col1, col2 = st.columns([1, 1])
with col1:
    start = st.button("Transcribe")
with col2:
    clear = st.button("Clear")

if clear:
    st.session_state.transcript = None
    st.session_state.meta = None
    st.experimental_rerun()

if start:
    if not url or not is_valid_youtube_url(url):
        st.error("Please enter a valid YouTube URL.")
    else:
        try:
            with st.spinner("Fetching video and extracting audio..."):
                filepath, meta = download_audio_from_youtube(url)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                st.info(
                    f"Selected audio stream: {meta.get('mime_type', 'unknown')}, "
                    f"{meta.get('abr', 'unknown bitrate')}, ~{size_mb:.2f} MB"
                )
                if size_mb > max_size_mb:
                    st.warning(
                        f"Audio file is {size_mb:.2f} MB which exceeds the configured limit of {max_size_mb} MB. "
                        "Please try a shorter video or reduce the max size if your API plan allows."
                    )
                    raise RuntimeError("Audio file exceeds configured size limit.")

            with st.spinner("Transcribing audio with OpenAI Whisper..."):
                transcript_text = transcribe_audio(filepath, language=lang)
                st.session_state.transcript = transcript_text
                st.session_state.meta = meta

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display results
if st.session_state.transcript:
    m = st.session_state.meta or {}
    st.subheader("Transcription Result")
    if m:
        st.write(
            f"Title: {m.get('title', 'Unknown')}\n"
            f"Author: {m.get('author', 'Unknown')}\n"
            f"Duration: {format_duration(m.get('length', 0))}\n"
            f"Views: {m.get('views', 'N/A')}\n"
            f"Published: {m.get('publish_date', 'N/A')}"
        )
    st.text_area("Transcript", st.session_state.transcript, height=300)
    st.download_button(
        label="Download Transcript (.txt)",
        data=st.session_state.transcript,
        file_name="transcript.txt",
        mime="text/plain"
    )