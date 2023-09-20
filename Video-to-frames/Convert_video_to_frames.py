# reference https://web.archive.org/web/20161010175545/https://tobilehman.com/blog/2013/01/20/extract-array-of-frames-from-mp4-using-python-opencv-bindings/

import cv2

# open the video file
vf = cv2.VideoCapture(
    "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/video_to_split/video.mp4")

# use the read method on the video capture object to return
# a tuple where the first element is the success flag and the second is the
# image array of rgb values
success, image = vf.read()

# convert the video to frames and save each one
# Loop runs until the flag is false
frame_count = 0
while success:
    success, image = vf.read()
    print(success)
    cv2.imwrite("C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/frames/frame%d.jpg" % frame_count, image)
    frame_count += 1
