# This program will feed in live test data (new persons face) and run that through the existing SVM model to detect a verified face
# WARNING--- for an image to be used in the SVM it must be loaded, converted into 100,100 image, converted to grayscale and flattened
# reshape image to (1,-1) as one sample is used
import cv2
import numpy as np
import os
import pickle
from PIL import Image
# import RPi.GPIO as GPIO
import time
# value for an authorized user
authorized = "authorized_user"
# value for unauthorized user
unauthorized = "unauthorized_user"

# add in some code to control the GPIO pin
led_pin = 12
led_interval = .1
# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)

# GPIO.output(led_pin, GPIO.HIGH)
# time.sleep(led_interval)
# GPIO.output(led_pin, GPIO.LOW)
# time.sleep(led_interval)
# GPIO.output(led_pin, GPIO.HIGH)
# time.sleep(led_interval)
# GPIO.output(led_pin, GPIO.LOW)
print("Indicated the 12th GPIO pin is working.....")

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
        photo_filename).convert('L'))
    cv2.imshow('img', opened_img)
    cv2.waitKey(0)

    # try and use the haarcascade filter on the image to better work with the model
    haar_cascade = cv2.CascadeClassifier(
        "XML/haarcascade_frontalface_default.xml")
    faces_rect = haar_cascade.detectMultiScale(
        opened_img, scaleFactor=1.1, minNeighbors=15)
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
        print(f"Prediction.....{prediction}")
        if (prediction == authorized):
            # blink the GPIO pin
            # GPIO.output(led_pin, GPIO.HIGH)
            # time.sleep(led_interval)
            # GPIO.output(led_pin, GPIO.LOW)
            # time.sleep(led_interval)
            # GPIO.output(led_pin, GPIO.HIGH)
            # time.sleep(led_interval)
            # GPIO.output(led_pin, GPIO.LOW)
            print("The light is blinking....successful unlock")
        else:
            print("the light is not on....door locked")
    else:
        print("Something went wrong, no face detected")


# run the functions to test
newPhotoFromCam()
getLabel()
