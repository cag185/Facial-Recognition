# this file should be called with an input param which will create a new folder to store in a video of the new user

import cv2
import os
from PIL import Image
# from picamera2 import Picamera2
import RPi.GPIO as GPIO
import sys
import time

# check if the API has called this
led_pin = 17
led_int = .5
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.output(led_pin, GPIO.HIGH)


def recordVideoLaptop(file_dest):
    print("inside function")
    # start a recording with CV2
    cap = cv2.VideoCapture('/dev/video0')
    # check if camera opened
    if not cap.isOpened():
        print("Error: could not open camera.")
        exit()

    # set features
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # video duration
    video_dur = 10

    # create a video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_dest, fourcc, 20.0, (width, height))

    # prepare to start recording
    print(".................................................................")
    print("The camera is about to start recording for 10 seconds in 3 seconds. Please stare at the camera lens")
    print(".................................................................")
    time.sleep(3)
    start_time = time.time()
    end_time = start_time + video_dur
    while time.time() < end_time:
        # capture each frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        # display frame
        # cv2.imshow('Frame', frame)

        # write to the video output
        out.write(frame)

    # Release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {file_dest}")

    # after recording, try to split to frames
    try:
        print("Converting video to frames...")
        import Convert_video_to_frames
    except Exception as e:
        print(f"Error converting video to frames: {e}")
    else:
        print("Success....")

# function to take a recording using the rasb pi cam


def recordVideoPi(file_dest):
    # set the GPIO pin for the recording LED
    recording_indicator = 17
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(recording_indicator, GPIO.OUT, initial=GPIO.LOW)

    # Initialize the camerea
    cam = Picamera2()
    cam.preview_configuration.main.size = (1920, 1080)
    cam.framerate = 60
    cam.preview_configuration.main.align()
    cam.configure("Preview")
    cam.start()

    # the camera is about to start recording
    print("....................................................")
    print("The camera is about to start recording for 10 seconds")
    print("....................................................")

    # set the LED indicator to high
    GPIO.output(recording_indicator, GPIO.HIGH)

    pic_count = 0
    frame_array = []

    # start the recording
    start_time = time.time()
    end_time = start_time + 10
    while (time.time() < end_time):
        frame_array.append(cam.capture_array())
    for frame in frame_array:
        pil_image = Image.fromarray(frame)
        pil_image_dest = file_dest + "face" + str(pic_count) + ".png"
        pil_image = pil_image.save(pil_image_dest)
        # finally increase the pic counter
        pic_count += 1

    GPIO.output(recording_indicator, GPIO.LOW)
    print(f"Frames saved to: {file_dest}")


# to get the first arg
user_to_create = sys.argv[1]  # gets the first argument
# user_to_create = "calebG"

print(f"The user to create {user_to_create}")

# # check if the folder does not exist currently
folder = "../Video-to-frames/video_to_split/"
folder_file = folder + user_to_create
filer = ".mp4"
# create the folder that points to the frame images
# folder = "../Video-to-frames/frames/"
# filer = user_to_create + "/"

# file_dest = os.path.join(folder, filer)
file_dest = folder_file + filer
doesExist = os.path.exists(file_dest)
if (doesExist):
    print(f"{user_to_create}.mp4 already exists!")
else:
    # # record the video
    # # on the laptop
    # # try to break into haar_cascade
    # create a destintation folder
    # os.mkdir(folder_file)
    print("Entering the recording prompt")
    recordVideoLaptop(file_dest)
    # recordVideoPi(file_dest)
    try:
        print("Converting frames to haar_cascade...")
        import FeatureExtraction
    except Exception as e:
        print(f"Error converting frames to haar_cascade: {e}")
    else:
        print("Success....")

    # try to train model
    try:
        print("Attempting to train model...")
        import OneClassSVM
    except Exception as e:
        print(f"Error training model: {e}")
    else:
        print("Success....")


time.sleep(led_int)
GPIO.output(led_pin, GPIO.LOW)
