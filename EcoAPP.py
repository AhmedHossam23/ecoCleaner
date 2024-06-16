import streamlit as st
import ultralytics
from ultralytics import YOLO
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase import firebase
import urllib.request
import numpy as np
import time
from PIL import Image
from io import BytesIO
import requests
from streamlit import session_state as state

# Initialize the model
model = YOLO("yolo-Weights/yolov8n.pt")

# Initialize Firebase
firebase = firebase.FirebaseApplication('https://khaled-ff982-default-rtdb.firebaseio.com/int', None)

# Class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Filter class names
filtered_classNames = ["cup", "bottle", "wine glass"]

# Function to load images from cam-URL
def load_from_esp_url(url):
    response = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(response.read()), dtype=np.uint8)
    return cv2.imdecode(imgnp, -1)

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Verify the content type
        content_type = response.headers.get('Content-Type')
        if not content_type or 'image' not in content_type:
            raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")

        # Attempt to open the image
        image = Image.open(BytesIO(response.content))
        return image

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching image from URL: {e}")
    except (IOError, ValueError) as e:
        raise RuntimeError(f"Error processing image data: {e}")

# Custom CSS for header with white and green theme
header_html = """
    <style>
    .header {
        font-size: 50px;
        text-align: center;
        color: white;
        padding: 100px;
        background: linear-gradient(to right, #6B8E23, #ADFF2F); /* Green shades */
        animation: fadeOut 5s forwards;
    }

    @keyframes fadeOut {
        0% {opacity: 1;}
        100% {opacity: 0;}
    }

    .hidden-content {
        display: none;
    }

    .show-content {
        display: block;
    }

    body {
        background-color: white;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #2E8B57; /* Dark green color */
    }

    .media-item {
        width: 800px; /* Set the width of all media items */
        height: 600px; /* Set the height of all media items */
        object-fit: cover; /* Ensure content fits within specified dimensions */
    }

    .image-caption, .video-caption {
        text-align: center;
        font-style: italic;
        color: #555555;
        margin-top: 5px;
    }

    .content-section {
        margin: 20px 0;
    }
    </style>
    <div class="header">
        Welcome to Eco Cleaner
    </div>
    <script>
    setTimeout(function(){
        document.querySelector('.header').style.display = 'none';
        document.querySelector('.content').classList.add('show-content');
    }, 5000); // 5 seconds delay
    </script>
"""

# Injecting the header HTML
st.markdown(header_html, unsafe_allow_html=True)

# Wait for the header animation to finish
time.sleep(5)

# Main content starts here
st.markdown('<div class="content hidden-content">', unsafe_allow_html=True)

st.title("Eco Cleaner")

# Project description
st.markdown("""
### Custom Object Detection App
This project demonstrates a real-time object detection system using an ESP-32 camera feed. The model detects objects in the video stream and displays bounding boxes around them.

#### Features:
- Real-time object detection with bounding boxes and labels.
- Interactive interface allowing users to start and stop the detection.
- Display of the camera feed along with detection results.
- Detailed project description with images and technology overview.

#### Technologies Used:
- **OpenCV**: For capturing video from the webcam and processing the frames.
- **Streamlit**: For creating the web application and displaying the video stream and results.
- **YOLOv8**: For providing the pre-trained object detection model.
""")

# Streamlit UI

# Paths to your images
image_paths = [
    "static/images/box.jpeg",
    "static/images/plastic.png",
    "static/images/detectionGlass.png"
]

# Captions for your images
image_captions = [
    "Our Box",
    "Plastic Detection",
    "Glass Detection"
]

# Paths to your videos
video_paths = [
    "static/videos/Design.mp4",
    "static/videos/mechanism.mp4"
]

# Display images in a single row
st.subheader("Images")
image_cols = st.columns(len(image_paths))
for col, img_path, caption in zip(image_cols, image_paths, image_captions):
    col.image(img_path, caption=caption, use_column_width=True)

# Display videos in a single row
st.subheader("Videos")
for video_path in video_paths:
    st.video(video_path, start_time=0)

st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state variables
if 'start_detection' not in state:
    state.start_detection = False
if 'stop_detection' not in state:
    state.stop_detection = False

# Add option to use the local camera
st.sidebar.header("Camera 📸 Settings and detection controller!")
local_camera = st.sidebar.checkbox("Use Local Camera")

# Add text input for camera IP when not using the local camera
camera_ip = ""
if not local_camera:
    camera_ip = st.sidebar.text_input("Enter Camera IP Address")

# Functions to handle detection and display
def detect_and_display(local_camera, cam_url):
    if local_camera:
        cap = cv2.VideoCapture(cam_url)
        cap.set(3, 640)
        cap.set(4, 480)
        while cap.isOpened() and state.start_detection:
            success, img = cap.read()
            if success:
                results = model(img, stream=True)
                process_results(results, img, local_camera)
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB")
            else:
                break
        cap.release()
    else:
        while state.start_detection:
            frame = load_from_esp_url(cam_url)
            if frame is not None:
                results = model(frame, stream=True)
                process_results(results, frame, local_camera)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB")

def process_results(results, img, local_camera):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            class_id = int(box.cls[0])
            class_name = classNames[class_id]

            if local_camera or class_name in filtered_classNames:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Default Camera URL
cam_url = camera_ip if camera_ip else "http://192.168.4.1/cam-hi.jpg"

# UI Elements to Start and Stop Detection
start_button = st.sidebar.button("Start Detection")
stop_button = st.sidebar.button("Stop Detection")

if start_button:
    state.start_detection = True
    stframe = st.empty()
    detect_and_display(local_camera, 0 if local_camera else cam_url)

if stop_button:
    state.start_detection = False

# # User Inputs for Waste Type Detection
# waste_types = ["Plastic", "Glass", "Metal"]
# selected_waste_type = st.sidebar.selectbox("Select Waste Type", waste_types)

# if selected_waste_type:
#     st.sidebar.write(f"Selected Waste Type: {selected_waste_type}")
