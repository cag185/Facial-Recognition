# this file will be used to test if the motion detector is working

import os
import RPi.GPIO as GPIO
import time
import ToggleLock

# set up the GPIO stuff
MotionDetector_Pin = 10
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(MotionDetector_Pin, GPIO.IN)
print("GPIO configured")

# create an instance of the ToggleLockClass
toggleLock = ToggleLock.ToggleLockClass()

# check for motion
try:
    print("Motion detector now armed....")
    while True:
        if GPIO.input(MotionDetector_Pin):
            print("---Motion Detected---")
            # run the classifier toggle
            toggleLock.callToggle()
            print("-----Done Predicting.------")
            time.sleep(5)  # sleep for two seconds to avoid repeat detection
    # time.sleep(5)

except KeyboardInterrupt:
    print("Closing...")
finally:
    GPIO.cleanup()
