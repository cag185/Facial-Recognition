# Want to take in 100 samples from each user and limit them using the
# feature extraction. Use this to then do PCA on and store
# from here we can train the model on less input and using only faces and see
# how to accuracy reflects this and the runtime


# Feature extraction using the haarcascade lib
import os
import cv2

# load in the haarcascade
haar_cascade = cv2.CascadeClassifier("XML/haarcascade_frontalface_default.xml")

# parent dir
parent_dir = "Video-to-frames/frames/"
person_folders = os.listdir(parent_dir)

# want to get the images in each persons directory
for person in person_folders:
    person_folder_path = parent_dir + person + "/"
    # get each picture in the persons folder
    pics = os.listdir(person_folder_path)
    pic_count = 0
    for pic in pics:
        # grab each pic
        while (pic_count <= 75):
            file_path = person_folder_path + pic
            img = cv2.imread(file_path)
            # convert the image to gray
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                cv2.imshow('og image', img)
            except Exception as e:
                print(f"exception {e}")

            try:
                # create the face rect
                faces_rect = haar_cascade.detectMultiScale(
                    gray_img, scaleFactor=1.01, minNeighbors=15)

                # Iterating through rectangles of detected faces
                # get just one face
                size_faces_array = len(faces_rect)
                # if (size_faces_array > 0):
                (x, y, w, h) = faces_rect[0]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.imshow(img)

                # Crop the image to include only the face
                face_roi = img[y:y+h, x:x+w]

                # save the cropped photos to an output folder
                outfolder = "Video-to-frames/haar_cascade_frames/"

                # check if the persons folder already exists
                person_out_folder = outfolder + person
                doesExist = os.path.exists(person_out_folder)
                if not doesExist:
                    os.makedirs(person_out_folder)

                file_dest = person_out_folder + "/" + \
                    "face_%d.jpg" % (pic_count+1)

                # save the file here with the jpg extension
                cv2.imwrite(file_dest, face_roi)
                pic_count += 1
            except Exception as e:
                print(f"error: {e}")
