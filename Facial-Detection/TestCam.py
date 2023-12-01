import cv2
from picamera2 import Picamera2, Preview
import time
import os

cam = Picamera2()
cam.preview_configuration.main.size = (1280, 720)
piCam.framerate = 500
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.main.align()
piCam.configure("preview")
piCam.start()

# grab a frame
print("prepate to take a pic.....")
time.sleep(1)
frame = piCam.capture_array()
gray = frame.convert('L')
new_test_img = "new_testing_img.png"
cv2.imwrite(new_test_img, gray)
