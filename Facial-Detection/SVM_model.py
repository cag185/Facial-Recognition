# this file will create a SVM machine based on the training data provided from the user faces' databank
# SVM will be trained based on PCA dim reduced images

from sklearn.svm import LinearSVC
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import cv2

# variable to store the training accuracy
training_acc = 0


class SVM_facial_detection():
    # import the data as an array
    training_acc = 0

    def __init__(self):
        self.training_acc = None

    def train_model(self):
        parent_folder = "../Facial-Profile-Databank/"
        face_list = os.listdir(parent_folder)
        # import array -- holds the flattened images
        flat_data_arr = []
        # label array -- holds the label of the images
        label_arr = []

        # iterating var
        person_label = 0
        # per each person that has a facial profile
        for person in face_list:
            print("loading....person: " + person)
            # for each image in that profile
            image_dir = parent_folder + person + "/"
            img_list = os.listdir(image_dir)

            for image in img_list:
                full_path = image_dir + image

                # open the image as an array
                curr_image = Image.open(full_path)
                curr_image_nparray = np.array(curr_image)
                # flatten the image
                flat_data_arr.append(curr_image_nparray.flatten())
                # append it to a flattened array
                label_arr.append(face_list[person_label])
            print("loaded person: " + person + " successfully")
            # at the end, add 1 to the iterator
            person_label += 1

            # at this point the two arrays from the start should be np arrays
            label_data = np.array(label_arr)
            print("done preprocessing")

            # convert the flat data array into np array
            flat_data = np.array(flat_data_arr)

            # create a pandas dataframe out of the data
            df = pd.DataFrame(flat_data)
            df['Label'] = label_data

            print(df.shape)
            print("done converting to data frame")

            # separate the input features and the labels from the dataframe
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1]

        # split the data into testing and training data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=77, stratify=y)

        # standardize the data
        scaler = StandardScaler()

        # xtrain and x_test
        x_train_fit = scaler.fit_transform(x_train)
        x_test_fit = scaler.fit_transform(x_test)

        # create the SVM classifier
        lsvc = svm.LinearSVC(dual=True, max_iter=10000)
        print("the classifier has been created")
        # fit the model to the data
        # lsvc.fit(x_train, y_train)
        lsvc.fit(x_train_fit, y_train)
        print("The classifier has been trained")

        self.lsvc = lsvc

        # # test the model using the testing data
        # this is for the whole data brought in
        # y_pred = (lsvc.predict(x_test))
        y_pred = (lsvc.predict(x_test_fit))
        # compare the actual vs the prediction
        # this is for the whole data brought in
        training_acc = accuracy_score(y_pred, y_test)
        print("The training accuracy for the data in the training mode: " + training_acc)
        # save the training accuracy
        self.training_acc = training_acc

        # once we have created the model, we want to save it
        with open('SVM_model_larger.pkl', 'wb') as file:
            pickle.dump(lsvc, file)

    # create the function for prediction/accuracy
    def predict(self, x_data):
        # load the model from storage
        with open('SVM_model_larger.pkl', 'rb') as file2:
            lsvc_new = pickle.load(file2)
            return lsvc_new.predict(x_data)


# create an instance of the object
svm_object = SVM_facial_detection()

# train the model
svm_object.train_model()

# once the model is trained can retrieve the accuracy
print("training accuracy:" + svm_object.training_acc + "accurate")

# function for loading in a file and converting it into side


def loadNewPhoto(new_photo):
    # change the photo to be an nparray
    opened_img = np.asarray(Image.open(new_photo).resize((500, 500)))
    # reshape
    # plt.imshow(opened_img)
    # plt.show()
    flattened_img = (opened_img.flatten()).reshape(1, -1)
    new_pred = svm_object.predict(flattened_img)
    # output the prediction
    print("The label for the input data was " + new_pred)


# loadNewPhoto(new_photo=new_photo)

# load in the testing images for Caleb
caleb_dir = "Facial-Recognition/Test-Images/Caleb"
caleb_files = os.listdir(caleb_dir)
print("------ Testing images for Caleb ------")
for img in caleb_files:
    full_file = caleb_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Cam
cam_dir = "Facial-Recognition/Test-Images/Cam"
cam_files = os.listdir(cam_dir)
print("------ Testing images for Cam ------")
for img in cam_files:
    full_file = cam_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Hudson
hudson_dir = "Facial-Recognition/Test-Images/Hudson"
hudson_files = os.listdir(hudson_dir)
print("------ Testing images for Hudson ------")
for img in hudson_files:
    full_file = hudson_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Lucas
lucas_dir = "Facial-Recognition/Test-Images/Lucas"
lucas_files = os.listdir(lucas_dir)
print("------ Testing images for Lucas ------")
for img in lucas_files:
    full_file = lucas_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for MATT
matt_dir = "Facial-Recognition/Test-Images/Matt"
matt_files = os.listdir(matt_dir)
print("------ Testing images for Matt ------")
for img in matt_files:
    full_file = matt_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for JACK
jack_dir = "Facial-Recognition/Test-Images/Jack"
jack_files = os.listdir(jack_dir)
print("------ Testing images for Jack ------")
for img in jack_files:
    full_file = jack_dir + "/" + img
    loadNewPhoto(full_file)
