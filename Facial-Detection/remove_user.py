#!/usr/bin/env python

# this file should be called with an input param which will delete the folders of that user
import os
import sys
import shutil

# boolean to control the retraining
didDelete = False

# get the arg
user_to_delete = sys.argv[1]
print(f"User to delete {user_to_delete}")

# go to the mp4 file and delete it
video_folder = "~/Desktop/Facial-Recognition/Video-to-frames/video_to_split/"
path_to_remove = video_folder + user_to_delete + ".mp4"
does_exist = os.path.exists(path_to_remove)
if (does_exist):
    os.remove(path_to_remove)
else:
    print("path not found, mp4 video not here")

# go to the haarcascade folder and delete it
haar_cascade_folder = "~/Desktop/Facial-Recognition/Video-to-frames/haar_cascade_frames/"
path_to_remove = haar_cascade_folder + user_to_delete
does_exist = os.path.exists(path_to_remove)
if (does_exist):
    shutil.rmtree(path_to_remove, ignore_errors=True)
    didDelete = True
else:
    print("path not found, haar_cascade folder not here")


# delete the individual frames
haar_cascade_folder = "~/Desktop/Facial-Recognition/Video-to-frames/frames/"
path_to_remove = haar_cascade_folder + user_to_delete
does_exist = os.path.exists(path_to_remove)
if (does_exist):
    shutil.rmtree(path_to_remove, ignore_errors=True)
else:
    print("path not found, face folder not here")

# finally call the function to retrain the model
if (didDelete):
    import OneClassSVM
