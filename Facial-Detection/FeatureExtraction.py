# Want to take in 100 samples from each user and limit them using the
# feature extraction. Use this to then do PCA on and store
# from here we can train the model on less input and using only faces and see
# how to accuracy reflects this and the runtime


# Feature extraction using the haarcascade lib
import os
import cv2

# load in the haarcascade
haar_cascade = cv2.CascadeClassifier(
    "../XML/haarcascade_frontalface_default.xml")

# parent dir
parent_dir = "../Video-to-frames/frames/"
person_folders = os.listdir(parent_dir)

# want to get the images in each persons directory
print("Loading in the files")
for person in person_folders:
    person_folder_path = parent_dir + person + "/"
    # check if the foler already exists so that we dont generate more files than needed
    folder = os.path.join(
        "../Video-to-frames/haar_cascade_frames/", person)
    folderExists = os.path.exists(folder)
    if (not folderExists):
        # generate the folder
        # os.mkdir(folder)
        # get each picture in the persons folder
        pics = os.listdir(person_folder_path)
        pic_count = 0
        # grab every picture to train a better model
        while (pic_count < len(pics)):
            # grab each pic
            file_path = person_folder_path + pics[pic_count]
            img = cv2.imread(file_path)
            # convert the image to gray
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            try:
                # create the face rect
                faces_rect = haar_cascade.detectMultiScale(
                    gray_img, scaleFactor=1.1, minNeighbors=15)

                # Iterating through rectangles of detected faces
                # get just one face
                size_faces_array = len(faces_rect)
                if (size_faces_array > 0):
                    (x, y, w, h) = faces_rect[0]
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # cv2.imshow('show', img)
                    # cv2.waitKey(0)

                    # Crop the image to include only the face
                    face_roi = img[y:y+h, x:x+w]

                    # save the cropped photos to an output folder
                    outfolder = "../Video-to-frames/haar_cascade_frames/"

                    # check if the persons folder already exists
                    person_out_folder = outfolder + person
                    doesExist = os.path.exists(person_out_folder)
                    if not doesExist:
                        os.mkdir(person_out_folder)

                    file_dest = person_out_folder + "/" + \
                        "face_%d.jpg" % (pic_count+1)

                    # save the file here with the jpg extension
                    cv2.imwrite(file_dest, face_roi)
            except Exception as e:
                print(f"error: {e}")
            pic_count += 1
print("Done")
