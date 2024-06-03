import streamlit as st
import ultralytics
from ultralytics import YOLO
import cv2
import math
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


# Function to load images from URL
# def load_image_from_url(url):
#     response = requests.get(url)
#     return Image.open(BytesIO(response.content))


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
- **YOLOv5**: For providing the pre-trained object detection model.
""")

# Streamlit UI

# Paths to your images
image_paths = [
    "ecoCleanerAPP/static/images/box.jpeg",
    "ecoCleanerAPP/static/images/plastic.png",
    "ecoCleanerAPP/static/images/detectionGlass.png"
]

# Captions for your images
image_captions = [
    "Our Box",
    "Plastic Detection",
    "Glass Detection"
]

# Paths to your videos
video_paths = [
    "/Users/user/Desktop/Work/RealTime/ecoCleanerAPP/static/videos/Design.mp4",
    "/Users/user/Desktop/Work/RealTime/ecoCleanerAPP/static/videos/mechanism.mp4"
]


st.title("Custom Object Detection App")

# Project description and examples 
st.write("""
This project demonstrates a real-time object detection system using a ESP-32 📸 Camera feed.
The model detects objects in the video stream and displays bounding boxes around them.

## Features:
- **Real-time** object detection with bounding boxes and labels.
- Interactive interface allowing users to start and stop the detection.
- Display of the camera feed along with detection results.
- Detailed project description with images and technology overview.

## Technologies Used:
- **OpenCV**: For capturing video from the webcam and processing the frames.
- **Streamlit**: For creating the web application and displaying the video stream and results.
- **YOLOv8**: For providing the pre-trained object detection model .

*Below are some examples of object detection in action:*
""")



# Display images in a single row
st.subheader("Images")
image_cols = st.columns(len(image_paths))
for col, img_path, caption in zip(image_cols, image_paths, image_captions):
    col.image(img_path, caption=caption, use_column_width=True)

# Display videos in a single row
st.subheader("Videos")
video_cols = st.columns(len(video_paths))
for col, video_path in zip(video_cols, video_paths):
    col.video(video_path, start_time=0)

st.markdown('</div>', unsafe_allow_html=True)






# Display example images in the description section
# example_image_urls = [
#     "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.linkedin.com%2Fpulse%2Fopencv-java-yolo-object-detection-images-svetozar-radoj%25C4%258Din&psig=AOvVaw3lZtKWIzU5Nyl2gc_dC0yz&ust=1717256856763000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCLCMn7meuIYDFQAAAAAdAAAAABAR" , # Replace with actual URLs of example images
#     "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FExample-of-Object-Detection-Bottle-Detection_fig4_341509210&psig=AOvVaw2Z3V5vXhRCeVwqvh8umj1c&ust=1717254510042000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCLjNrtqVuIYDFQAAAAAdAAAAABAJ"   # Replace with actual URLs of example images
# ]
# for url in example_image_urls:
#     try:
#         image = load_image_from_url(url)
#         st.image(image, caption="Example Object Detection", use_column_width=True)
#     except RuntimeError as e:
#         st.error(f"Failed to load image from URL: {e}")

# show the images and videos in streamlit UI

# note that the directories is suitable for production not local host
# st.image("/Users/user/Desktop/Work/RealTime/ecoCleanerAPP/static/images/box.jpeg", caption="Our Box", use_column_width=True)
# st.image("/Users/user/Desktop/Work/RealTime/ecoCleanerAPP/static/images/plastic.png", caption="Our Box", use_column_width=True)
# st.image("/Users/user/Desktop/Work/RealTime/ecoCleanerAPP/static/images/detectionGlass.png", caption="Our Box", use_column_width=True)
# st.video("/Users/user/Desktop/Work/RealTime/ecoCleanerAPP/static/videos/Design.mp4", start_time=0)
# st.video("/Users/user/Desktop/Work/RealTime/ecoCleanerAPP/static/videos/mechanism.mp4", start_time=0)




st.sidebar.header("ESP-32 📸 Camera Settings and detection controller!")
cam_url = st.sidebar.text_input("Camera IP:", "http://192.168.1.10/cam-hi.jpg")


# Initialize session state variables
if 'start_detection' not in state:
    state.start_detection = False
if 'stop_detection' not in state:
    state.stop_detection = False



# Initialize Firebase
firebase = firebase.FirebaseApplication('https://khaled-ff982-default-rtdb.firebaseio.com/int', None)

# Function to detect objects
def detect_objects(cam_url):
    stframe = st.empty()
    
    while state.start_detection:
        frame = load_from_esp_url(cam_url)
        
        if frame is not None:
            results = model(frame, stream=True)
            firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 0)
            

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if classNames[int(box.cls[0])] in filtered_classNames:
                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        if confidence > 0.6 and classNames[cls] in filtered_classNames:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            cv2.putText(frame, f"{classNames[cls]}: {confidence}", org, font, fontScale, color, thickness)
                            # hardware signal control through firebase
                            if classNames[cls] == 'bottle':
                                firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 2)
                                time.sleep(2)
                                firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 0)
                            elif classNames[cls] == 'cup' or classNames[cls] == 'wine glass':
                                firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 1)
                                time.sleep(3)
                                firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 0)
                            else:
                                firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 0)
                            

            # Convert frame to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

        if state.stop_detection:
            break

# Function to display the camera stream continuously
def display_camera_stream(cam_url):
    stframe = st.empty()
    while not state.start_detection:
        frame = load_from_esp_url(cam_url)
        if frame is not None:
            # Convert frame to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")


# Buttons to start and stop detection
if st.sidebar.button("Start Detection"):
    state.start_detection = True
    state.stop_detection = False

if st.sidebar.button("Stop Detection"):
    state.stop_detection = True
    state.start_detection = False

# Control the stream based on the state
if state.start_detection:
    detect_objects(cam_url)
else:
    if state.stop_detection:
        st.write("Detection stopped. Press 'Start Detection' to resume.")

