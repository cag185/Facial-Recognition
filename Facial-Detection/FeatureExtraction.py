# Feature extraction using the haarcascade lib
import os
import cv2

# load in the haarcascade
haar_cascade = cv2.CascadeClassifier("XML/haarcascade_frontalface_default.xml")

# parent dir
parent_dir = "Test-Images/Hudson/"
pics = os.listdir(parent_dir)

for pic in pics:
    file_path = parent_dir + pic
    img = cv2.imread(file_path)
    # convert the image to gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # create the face rect
    faces_rect = haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.01, minNeighbors=9)

    # Iterating through rectangles of detected faces
    # get just one face
    (x, y, w, h) = faces_rect[0]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Crop the image to include only the face
    face_roi = img[y:y+h, x:x+w]

    cv2.imshow('Original Image', img)
    try:
        cv2.imshow('Cropped Face', face_roi)
    except Exception as e:
        print(f"Error: {e}")

    cv2.waitKey(0)

    # save the cropped photos to an output folder
    outfolder = "Video-to-frames/haar_cascade_frames/"
