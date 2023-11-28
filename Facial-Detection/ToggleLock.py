# This program will feed in live test data (new persons face) and run that through the existing SVM model to detect a verified face
# WARNING--- for an image to be used in the SVM it must be loaded, converted into 100,100 image, converted to grayscale and flattened
# reshape image to (1,-1) as one sample is used
import cv2
import numpy as np
import os
import pickle
from PIL import Image

# load in the SVC model
with open('SVC_model_larger.pkl', 'rb') as file:
    lsvc = pickle.load(file)

# create a function to load in a new image
photo_filename = "testphoto.png"


def newPhotoFromCam():
    # take the photo
    cam_index = 0
    print("waiting to take picture, press space to take")
    key = cv2.waitKey(0)
    capture = cv2.VideoCapture(cam_index)
    if not capture.isOpened():
        print("Error: could not open webcam")
        return
    ret, frame = capture.read()
    if not ret:
        print("Error: failed to capture a frame.")
        capture.release()
        return

    cv2.imwrite(photo_filename, frame)
    cv2.imshow('image', frame)
    # release webcam
    capture.release()
    cv2.destroyAllWindows()


def getLabel():
    opened_img = np.asarray(Image.open(
        photo_filename).convert('L').resize((100, 100)))
    new_image = (opened_img.flatten()).reshape(1, -1)
    prediction = lsvc.predict(new_image)
    print(f"Prediction.....{prediction}")


# run the functions to test
newPhotoFromCam()
getLabel()
