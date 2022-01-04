# Imports
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from time import sleep as te
import datetime
from datetime import date
from datetime import *

# Defining the path no mask directory
no_mask_dir_path = "No Mask People"
parent_directory = "/Users/yashsolanki/Desktop/Neural Networks/live-face-mask-detector"

# Defining the class names
class_names = ['mask', 'no mask']

# Getting the current time
time_right_now = datetime.now()

# Getting the current formatted date
date_today = date.today()

no_mask_index = 0
countdown_initial = 150

# Loading in the face casecade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading in the model
model = load_model("less-data-mask-model-v1.0/")

# Creating the countdown


def countdown_timer(t):
    mins, secs = divmod(t, 60)
    timer = '{:02d}:{:02d}'.format(mins, secs)
    print(timer, end="\r")
    te.sleep(1)
    t -= 1

# Getting the encoded faces


def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                face = fr.load_image_file("No Mask People/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    print(encoded)

    return encoded


# Getting the video feed
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    label = "No Detection Done Till Now"
    faces_list = []
    preds = []
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (300, 300))
        # face_frame = image.load_img(face_frame, target_size=(300, 300))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=[0])
        # face_frame = preprocess_input(face_frame)
        face_frame = np.vstack([face_frame])
        faces_list.append(face_frame)
        # if len(faces_list) > 0:
        #     preds = model.predict(faces_list)

        preds = model.predict(face_frame)
        # color = (0, 255, 0) if preds[0][0] == 0  else (0, 0, 255)

        # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        label = class_names[np.argmax(preds)]
        label = "{}: {:.2f}%".format(
            label, max(preds[0][0], preds[0][1]) * 100)

        if label == "mask":
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        cv2.putText(frame, label, (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (color), 2)

        # print("Detection Done pls wait for 5 seconds for another detection")
        # countdown_timer(5)
        # te.sleep(5)
        if label == "mask":
            print('MASK')
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

           # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

           # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        else:
            print('NO MASK')
            # while (video_capture.isOpened()):

            # Taking a screenshot of the screen evry frame

            try:
                no_mask_path = os.path.join(
                    parent_directory, no_mask_dir_path)
                os.mkdir(no_mask_path)
                cv2.imwrite("No Mask People/" + str(time_right_now) + "%d.jpg" %
                            no_mask_index, frame)
                no_mask_index += 1

            except Exception as e:
                cv2.imwrite("No Mask People/" + str(time_right_now) + "%d.jpg" %
                            no_mask_index, frame)
                no_mask_index += 1

            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
