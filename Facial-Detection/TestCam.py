import cv2
from camera2 import camera2, Preview
import time
import os

cam = camera2()
cam.preview_configuration.main.size = (1280, 720)
cam.framerate = 500
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.main.align()
cam.configure("preview")
cam.start()

# grab a frame
print("prepate to take a pic.....")
time.sleep(1)
frame = cam.capture_array()
gray = frame.convert('L')
new_test_img = "new_testing_img.png"
cv2.imwrite(new_test_img, gray)
