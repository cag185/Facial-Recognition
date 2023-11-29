# this file will create a one class classifier based on the user face images

import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import OneClassSVM
from sklearn import svm
import time


class SVC_facial_detection():
    def __init__(self):
        self.testing_accuracy = None

    def train_model(self):
        # try loading in the non compressed images
        # parent_folder = "Facial-Profile-Databank/"
        # parent_folder = "Video-to-frames/haar_cascade_frames/"
        parent_folder = "../Video-to-frames/haar_cascade_frames/"
        face_list = os.listdir(parent_folder)
        flat_data_arr = []
        label_arr = []

        # iterating variable
        person_label = 0

        # loop through each person and give them an authorized label
        for person in face_list:
            print("loading...person: " + person)
            image_dir = parent_folder + person + "/"
            img_list = os.listdir(image_dir)

            # for each image that belongs to that person
            img_count = 0
            for image in img_list:
                if (img_count >= 300):
                    break
                full_path = image_dir + image
                curr_image = Image.open(full_path)
                curr_image = curr_image.convert('L')
                # resize the images

                curr_image_nparray = np.array(curr_image)
                curr_image_nparray = cv2.resize(curr_image_nparray, (100, 100))

                # flatten the image
                flat_data_arr.append(curr_image_nparray.flatten())
                # append to the flattened label array
                # label_arr.append(face_list[person_label])
                if (person == "unauthorized_users"):
                    label_arr.append(-1)
                else:
                    label_arr.append(1)
                img_count += 1

            print("loaded person: " + person + " successfully")
            person_label += 1

        # now that we are done collecting images of people
        label_data = np.array(label_arr)
        # flat_data = np.array(flat_data_arr)
        flat_data = np.array(flat_data_arr)
        print("done preprocessing")

        # separate the input features and labels
        x = flat_data
        y = label_data

        # split the data into training and testing data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42)

        # create the classifier
        OCSVM = OneClassSVM(kernel="linear", nu=0.05)
        print("Done creating the OCSVM")

        # fit the trained model to the data
        print("Training Model....")
        start_time = time.time()
        OCSVM.fit(x_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Done Training Model in " + str(elapsed_time) + " seconds")
        y_pred = (OCSVM.predict(x_test))

        self.testing_accuracy = accuracy_score(y_pred, y_test)
        print("Testing accuracy score: " + str(self.testing_accuracy))
        with open('OCSVM_model.pkl', 'wb') as file:
            pickle.dump(OCSVM, file)

    # create the function for prediction/accuracy

    def predict(self, x_data):
        # load the model from storage
        with open('../OCSVM_model.pkl', 'rb') as file2:
            lsvc_new = pickle.load(file2)
            return str(lsvc_new.predict(x_data))


def loadNewPhoto(new_photo):
    # change the photo to be cv2 in split colors
    # img = cv2.cvtColor(cv2.imread(new_photo), cv2.COLOR_BGR2RGB)
    img = np.array(Image.open(new_photo).convert('L'))
    full_image = cv2.resize(img, (100, 100))
    # split rgb
    # blue, green, red = cv2.split(full_image)
    # norm_blue = blue/255
    # norm_green = green/255
    # norm_red = red/255

    # pca_b = PCA(n_components=.99)
    # pca_b.fit(norm_blue)
    # trans_pca_b = pca_b.transform(norm_blue)

    # pca_g = PCA(n_components=.99)
    # pca_g.fit(norm_green)
    # trans_pca_g = pca_g.transform(norm_green)

    # pca_r = PCA(n_components=.99)
    # pca_r.fit(norm_red)
    # trans_pca_r = pca_r.transform(norm_red)

    # recombine the images
    # b_arr = pca_b.inverse_transform(trans_pca_b)
    # g_arr = pca_g.inverse_transform(trans_pca_g)
    # r_arr = pca_r.inverse_transform(trans_pca_r)

    # merge the photos back to one
    # img_reduced = (cv2.merge((r_arr, g_arr, b_arr)))
    # flattened_img = (img_reduced.flatten()).reshape(1, -1)
    flattened_img = full_image.flatten().reshape(1, -1)
    # need to also compress the image using PCA
    new_pred = svc_object.predict(flattened_img)
    # output the prediction
    print("The label for the input data was " + new_pred)


# create an instance of the object
svc_object = SVC_facial_detection()

# train the object
svc_object.train_model()

# load in the testing images for Caleb
# caleb_dir = "Facial-Recognition/Test-Images/Caleb"
caleb_dir = "Test-Images/Caleb"
caleb_files = os.listdir(caleb_dir)
print("------ Testing images for Caleb ------")
for img in caleb_files:
    full_file = caleb_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Cam
cam_dir = "../Facial-Recognition/Test-Images/Cam"
# cam_dir = "Test-Images/Cam"
cam_files = os.listdir(cam_dir)
print("------ Testing images for Cam ------")
for img in cam_files:
    full_file = cam_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Hudson
hudson_dir = "../Facial-Recognition/Test-Images/Hudson"
# hudson_dir = "Test-Images/Hudson"
hudson_files = os.listdir(hudson_dir)
print("------ Testing images for Hudson ------")
for img in hudson_files:
    full_file = hudson_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Lucas
lucas_dir = "../Facial-Recognition/Test-Images/Lucas"
# lucas_dir = "Test-Images/Lucas"
lucas_files = os.listdir(lucas_dir)
print("------ Testing images for Lucas ------")
for img in lucas_files:
    full_file = lucas_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for MATT
matt_dir = "../Facial-Recognition/Test-Images/Matt"
# matt_dir = "Test-Images/Matt"
matt_files = os.listdir(matt_dir)
print("------ Testing images for Matt ------")
for img in matt_files:
    full_file = matt_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for JACK
jack_dir = "../Facial-Recognition/Test-Images/Jack"
# jack_dir = "Test-Images/Jack"
jack_files = os.listdir(jack_dir)
print("------ Testing images for Jack ------")
for img in jack_files:
    full_file = jack_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Unathorized testing people
unauthor_dir = "../Test-Images/Unauthorized_users"
unauthor_files = os.listdir(unauthor_dir)
print("------ Testing images for Unauthorized Users ------")
for img in unauthor_files:
    full_file = unauthor_dir + "/" + img
    loadNewPhoto(full_file)
