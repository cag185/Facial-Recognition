import cv2
from PIL import Image
from picamera2 import Picamera2, Preview
import time
import os

cam = Picamera2()
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
pil_image = Image.fromarray(frame)
gray = pil_image.convert('L')
new_test_img = "new_testing_img.png"
gray.save(new_test_img)
cam.stop()
