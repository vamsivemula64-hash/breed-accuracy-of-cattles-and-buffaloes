import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

# Load YOLO model
st.title("YOLO Video Object Detection Prototype")
st.write("Upload a video and see detections!")

uploaded_model = st.file_uploader("Upload YOLO .pt model", type=["pt"])
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_model and uploaded_video:
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
        tmp_model.write(uploaded_model.read())
        model_path = tmp_model.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_video.read())
        video_path = tmp_video.name

    # Load model
    model = YOLO(model_path)

    # Process video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "output_annotated.mp4"
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    st.success("✅ Processing Complete!")
    st.video(output_path)
