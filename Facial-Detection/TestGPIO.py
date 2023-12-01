import cv2
import numpy as np
import os
import pickle
from PIL import Image
import RPi.GPIO as GPIO
import time

# define constants
unauthorized = 0
authorized = 1

# add in some code to control the GPIO pin
led_pin = 12
led_interval = .1
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)

GPIO.output(led_pin, GPIO.HIGH)
time.sleep(led_interval)
GPIO.output(led_pin, GPIO.LOW)
time.sleep(led_interval)
GPIO.output(led_pin, GPIO.HIGH)
time.sleep(led_interval)
GPIO.output(led_pin, GPIO.LOW)
print("Indicated the 12th GPIO pin is working.....")

# load in a testing image
print("loading images")
verified_user_photo = "../testphoto_verified.png"
unverified_user_photo = "../testphoto_unverified.png"

print("loading model")
# load in the svm model
with open('OCSVM_model.pkl', 'rb') as file:
    lsvc = pickle.load(file)

print("getting label from svm")


def getLabel(photo):
    opened_img = np.asarray(Image.open(
        photo).convert('L'))
    # cv2.imshow('img', opened_img)
    # cv2.waitKey(0)

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
            # blink the GPIO pin
            GPIO.output(led_pin, GPIO.HIGH)
            time.sleep(led_interval)
            GPIO.output(led_pin, GPIO.LOW)
            time.sleep(led_interval)
            GPIO.output(led_pin, GPIO.HIGH)
            time.sleep(led_interval)
            GPIO.output(led_pin, GPIO.LOW)
            print("The light is blinking....successful unlock")
        else:
            print("the light is not on....door locked")
    else:
        print("Something went wrong, no face detected")


# unlock on the verified user
print("Unlock verified user")
getLabel(verified_user_photo)
# remain locked on the non-verified user
print("Lock unverified user")
getLabel(unverified_user_photo)
