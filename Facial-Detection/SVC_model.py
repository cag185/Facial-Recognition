# this file will create a Linear SVC based on the user face images
# SVC will use PCA reduced images

import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn import svm
import time


class SVC_facial_detection():
    def __init__(self):
        self.training_acc = None

    def train_model(self):
        parent_folder = "Facial-Profile-Databank/"
        face_list = os.listdir(parent_folder)
        flat_data_arr = []
        label_arr = []

        # iterating variable
        person_label = 0

        # loop through each person that has a profile database and add them
        for person in face_list:
            # skip the not a person class
            if person == "unauthorized_users":
                break

            print("loading...person: " + person)
            image_dir = parent_folder + person + "/"
            img_list = os.listdir(image_dir)

            # for each image that belongs to that person
            for image in img_list:
                full_path = image_dir + image
                curr_image = Image.open(full_path)
                curr_image_nparray = np.array(curr_image)
                # flatten the image
                flat_data_arr.append(curr_image_nparray.flatten())
                # append to the flattened label array
                label_arr.append(face_list[person_label])
            print("loaded person: " + person + " successfully")
            person_label += 1

        # now that we are done collecting images of people
        label_data = np.array(label_arr)
        flat_data = np.array(flat_data_arr)
        print("done preprocessing")

        # create a pandas dataframe of the data
        # df = pd.DataFrame(flat_data)
        # df['Label'] = label_data
        # print("Done converting to data frame")
        # separate the input features and labels
        x = flat_data
        y = label_data

        # standardize the data
        # x = x / 255
        # split the data into training and testing data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42)

        # create the classifier
        # svc = svm.LinearSVC(dual="auto", max_iter=1000, random_state=42)
        svc = svm.SVC(kernel="linear", C=2.0)
        print("Done creating the SVC")

        # train the SVC with parameter tuning
        # param_dist = {'C': np.logspace(-3, 3, 7), }  # range for C

        # create the randomSearchCV object
        # print the machine is tuning and record how long
        # print("Tuning hyper-params with RandomSearchCV...")
        # start_time = time.time()
        # random_search = RandomizedSearchCV(
        #     svc, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, random_state=42
        # )
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print("Done tuning hyper-params in " + str(elapsed_time) + " seconds")

        # fit the trained model to the data
        print("Training Model....")
        start_time = time.time()
        # random_search.fit(x_train, y_train)
        svc.fit(x_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Done Training Model in " + str(elapsed_time) + " seconds")

        # best_estimator = random_search.best_estimator_
        # best_params = random_search.best_params_
        # print("Best params: " + str(best_params))

        # save the best estimator to be used wherever
        # self.best_model = best_estimator
        y_pred = (svc.predict(x_test))

        self.training_accuracy = accuracy_score(y_pred, y_test)
        print("Training accuracy score: " + str(self.training_accuracy))
        with open('SVC_model_larger.pkl', 'wb') as file:
            pickle.dump(svc, file)

    # create the function for prediction/accuracy

    def predict(self, x_data):
        # load the model from storage
        with open('SVC_model_larger.pkl', 'rb') as file2:
            lsvc_new = pickle.load(file2)
            return str(lsvc_new.predict(x_data))


def loadNewPhoto(new_photo):
    # change the photo to be cv2 in split colors
    img = cv2.cvtColor(cv2.imread(new_photo), cv2.COLOR_BGR2RGB)
    full_image = cv2.resize(img, (500, 500))
    # split rgb
    blue, green, red = cv2.split(full_image)
    norm_blue = blue/255
    norm_green = green/255
    norm_red = red/255

    pca_b = PCA(n_components=.99)
    pca_b.fit(norm_blue)
    trans_pca_b = pca_b.transform(norm_blue)

    pca_g = PCA(n_components=.99)
    pca_g.fit(norm_green)
    trans_pca_g = pca_g.transform(norm_green)

    pca_r = PCA(n_components=.99)
    pca_r.fit(norm_red)
    trans_pca_r = pca_r.transform(norm_red)

    # recombine the images
    b_arr = pca_b.inverse_transform(trans_pca_b)
    g_arr = pca_g.inverse_transform(trans_pca_g)
    r_arr = pca_r.inverse_transform(trans_pca_r)

    # merge the photos back to one
    img_reduced = (cv2.merge((r_arr, g_arr, b_arr)))
    flattened_img = (img_reduced.flatten()).reshape(1, -1)

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
# cam_dir = "Facial-Recognition/Test-Images/Cam"
cam_dir = "Test-Images/Cam"
cam_files = os.listdir(cam_dir)
print("------ Testing images for Cam ------")
for img in cam_files:
    full_file = cam_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Hudson
# hudson_dir = "Facial-Recognition/Test-Images/Hudson"
hudson_dir = "Test-Images/Hudson"
hudson_files = os.listdir(hudson_dir)
print("------ Testing images for Hudson ------")
for img in hudson_files:
    full_file = hudson_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Lucas
# lucas_dir = "Facial-Recognition/Test-Images/Lucas"
lucas_dir = "Test-Images/Lucas"
lucas_files = os.listdir(lucas_dir)
print("------ Testing images for Lucas ------")
for img in lucas_files:
    full_file = lucas_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for MATT
# matt_dir = "Facial-Recognition/Test-Images/Matt"
matt_dir = "Test-Images/Matt"
matt_files = os.listdir(matt_dir)
print("------ Testing images for Matt ------")
for img in matt_files:
    full_file = matt_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for JACK
# jack_dir = "Facial-Recognition/Test-Images/Jack"
jack_dir = "Test-Images/Jack"
jack_files = os.listdir(jack_dir)
print("------ Testing images for Jack ------")
for img in jack_files:
    full_file = jack_dir + "/" + img
    loadNewPhoto(full_file)
