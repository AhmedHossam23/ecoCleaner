import ultralytics
from ultralytics import YOLO
import cv2
import math
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import firebase
from firebase import firebase
import requests
import urllib.request
import numpy as np
import time


# Initialize the model
model = YOLO("yolo-Weights/yolov8n.pt")
cam_url = "http://192.168.1.39/cam-hi.jpg"


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

cap = cv2.VideoCapture(cam_url)
cap.set(3, 640)
cap.set(4, 480)

firebase = firebase.FirebaseApplication('https://khaled-ff982-default-rtdb.firebaseio.com/int', None)
while True: 

    # success, img = cap.read()
    # results = model(img, stream=True)


    img_resp = urllib.request.urlopen(cam_url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    results = model(img, stream=True)

    final_results = firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 0)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            if classNames[int(box.cls[0])] in filtered_classNames:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                # print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                if confidence > 0.6:
                    cv2.putText(img, f"{classNames[cls]}: {confidence}", org, font, fontScale, color, thickness)

                    # firebase
                    if classNames[cls] == 'bottle':

                        final_results = firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 2)

                        # firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int/", None, 2)
                        time.sleep(2)
                        final_results = firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 0)


                    elif classNames[cls] == 'cup' or classNames[cls] == 'wine glass':
                        final_results = firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 1)

                        # firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int/", None, 1)
                        time.sleep(3)
                        final_results = firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 0)
                    else:
                        final_results = firebase.put("https://khaled-ff982-default-rtdb.firebaseio.com/int", "Ahmed", 0)

                        # firebase.delete("https://khaled-ff982-default-rtdb.firebaseio.com/int/", None)
                    print(final_results)
                    # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)


    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



