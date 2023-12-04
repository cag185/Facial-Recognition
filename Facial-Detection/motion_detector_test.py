# this file will be used to test if the motion detector is working

import os
import RPi.GPIO as GPIO
import time

# set up the GPIO stuff
MotionDetector_Pin = 10
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(MotionDetector_Pin, GPIO.IN)
print("GPIO configured")

# check for motion
try:
    print("Motion detector now armed....")
    while True:
        if GPIO.input(MotionDetector_Pin):
            print("---Motion Detected---")
            # run the classifier toggle
            import ToggleLock
            time.sleep(2)  # sleep for two seconds to avoid repeat detection
    time.sleep(0.1)

except KeyboardInterrupt:
    print("Closing...")
finally:
    GPIO.cleanup()
