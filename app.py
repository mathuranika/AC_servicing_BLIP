import streamlit as st
st.set_page_config(page_title="AC Installation Captioning", layout="centered")

import os
import tempfile
import subprocess
import torch
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# ---------------- BLIP Model Setup ----------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval().to("cpu")
    return processor, model

processor, model = load_model()


# ---------------- OpenCV Scene Detection ----------------
def detect_scenes_opencv(video_path, output_dir, diff_threshold=30.0, min_scene_length=15):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    last_frame = None
    scenes = []
    current_frame = 0
    last_scene_change = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if last_frame is not None:
            diff = cv2.absdiff(gray, last_frame)
            score = diff.mean()

            if score > diff_threshold and (current_frame - last_scene_change) > min_scene_length:
                start_time = last_scene_change / fps
                end_time = current_frame / fps
                scenes.append((start_time, end_time))

                mid_frame = int((last_scene_change + current_frame) / 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                _, thumb = cap.read()
                thumb_path = os.path.join(output_dir, f"scene-{len(scenes):03d}.jpg")
                cv2.imwrite(thumb_path, thumb)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                last_scene_change = current_frame

        last_frame = gray
        current_frame += 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if last_scene_change < total_frames:
        start_time = last_scene_change / fps
        end_time = total_frames / fps
        scenes.append((start_time, end_time))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((last_scene_change + total_frames) / 2))
        _, thumb = cap.read()
        thumb_path = os.path.join(output_dir, f"scene-{len(scenes):03d}.jpg")
        cv2.imwrite(thumb_path, thumb)

    cap.release()
    return scenes


# ---------------- Captioning ----------------
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# ---------------- Subtitle Utilities ----------------
def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

def create_srt_file(scenes, captions, srt_path):
    with open(srt_path, 'w') as f:
        for idx, ((start, end), caption) in enumerate(zip(scenes, captions), 1):
            start_ts = format_timestamp(start)
            end_ts = format_timestamp(end)
            f.write(f"{idx}\n{start_ts} --> {end_ts}\n{caption}\n\n")


# ---------------- FFmpeg Burn-in ----------------
def burn_subtitles_ffmpeg(video_path, srt_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-vf",
        f"subtitles={srt_path}",
        "-c:a", "copy", output_path
    ], check=True)


# ---------------- Streamlit App UI ----------------
st.title("AC Installation Video Captioning")
st.write("Upload a video of an AC installation or service. The system will detect changes, generate scene-based captions, and return a captioned video.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        st.write("Processing the video. This may take a few minutes depending on its length.")

        video_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        frame_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        scenes = detect_scenes_opencv(video_path, frame_dir)

        captions = []
        for i in range(len(scenes)):
            thumb_path = os.path.join(frame_dir, f"scene-{i+1:03d}.jpg")
            caption = generate_caption(thumb_path)
            captions.append(caption)
            st.write(f"Scene {i+1}: {caption}")

        srt_path = os.path.join(tmp_dir, "captions.srt")
        create_srt_file(scenes, captions, srt_path)

        output_path = os.path.join(tmp_dir, "captioned_output.mp4")
        burn_subtitles_ffmpeg(video_path, srt_path, output_path)

        st.success("Captioned video is ready.")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("Download Captioned Video", f, file_name="captioned_output.mp4")
