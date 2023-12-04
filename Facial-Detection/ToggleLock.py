# This program will feed in live test data (new persons face) and run that through the existing SVM model to detect a verified face
# WARNING--- for an image to be used in the SVM it must be loaded, converted into 100,100 image, converted to grayscale and flattened
# reshape image to (1,-1) as one sample is used
import cv2
import numpy as np
import os
import pickle
from PIL import Image
# from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time
# value for an authorized user
authorized = 1
# value for unauthorized user
unauthorized = -1

# add in some code to control the GPIO pin (18)
led_pin = 18
led_interval = .5
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)
print("Indicated the 12th GPIO pin is working.....")


# load in the SVC model
with open('OCSVM_model.pkl', 'rb') as file:
    lsvc = pickle.load(file)

# initialize the camera
cap = cv2.VideoCapture('/dev/video2')
# check if cam exposed
if not (cap.isOpened()):
    print("Error cam not opened")
    exit()
width = 640
height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# state the testing image name
new_test_img = "new_testing_img.png"


def newPhotoFromCam():
    # take the photo
    print("Prepare to take a picture in 2 seconds")
    time.sleep(2)
    ret, frame = cap.read()
    pil_image = Image.fromarray(frame)
    pil_image.save(new_test_img)


def getLabel():
    opened_img = np.asarray(Image.open(
        new_test_img).convert('L'))

    # try and use the haarcascade filter on the image to better work with the model
    haar_cascade = cv2.CascadeClassifier(
        "../XML/haarcascade_frontalface_default.xml")
    faces_rect = haar_cascade.detectMultiScale(
        opened_img, scaleFactor=1.05, minNeighbors=15)
    # get just one face
    size_faces_array = len(faces_rect)
    print(f"size of face array: {size_faces_array}")
    if (size_faces_array > 0):
        (x, y, w, h) = faces_rect[0]
        cv2.rectangle(opened_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Crop the image to include only the face
        face_roi = opened_img[y:y+h, x:x+w]
        print("Face detected....")

        new_image = (cv2.resize(face_roi, (100, 100)).flatten()).reshape(1, -1)
        prediction = lsvc.predict(new_image)
        print(f"Prediction.....{prediction[0]}")
        if (prediction[0] == authorized):
            GPIO.output(led_pin, GPIO.HIGH)
            time.sleep(5)
            GPIO.output(led_pin, GPIO.LOW)
            print("The light is blinking....successful unlock")
        else:
            print("the light is not on....door locked")
    else:
        print("Something went wrong, no face detected")


# run the functions to test
newPhotoFromCam()
getLabel()

cam.stop()
