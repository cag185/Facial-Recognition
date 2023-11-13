# this file will create a SVM machine based on the training data provided from the user faces' databank
# SVM will be trained based on PCA dim reduced images

from sklearn.svm import LinearSVC
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import cv2

# create a class for the SVM model
# on init, the class will load the data and train the model


class SVM_facial_detection():
    # import the data as an array
    def train_model(self):
        parent_folder = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Facial-Profile-Databank/"
        face_list = os.listdir(parent_folder)
        # import array -- holds the flattened images
        flat_data_arr = []
        # label array -- holds the label of the images
        label_arr = []

        # iterating var
        person_label = 0
        # per each person that has a facial profile
        for person in face_list:
            print(f"loading....person: {person}")
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
            print(f"loaded person: {person} successfully")
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
        # assing this data to self
        # self.x_train = x_train
        # self.x_test = x_test
        # self.y_train = y_train
        # self.y_test = y_test

        # create the SVM classifier
        lsvc = svm.LinearSVC(verbose=1, dual="auto", max_iter=10000)
        print("the classifier has been created")
        # fit the model to the data
        (lsvc.fit(x_train, y_train))
        print("The classifier has been trained")

        self.lsvc = lsvc

        # # test the model using the testing data
        # this is for the whole data brought in
        y_pred = (lsvc.predict(x_test))
        # compare the actual vs the prediction
        # this is for the whole data brought in
        accuracy = accuracy_score(y_pred, y_test)

        # once we have created the model, we want to save it
        with open('SVM_model.pkl', 'wb') as file:
            pickle.dump(lsvc, file)

    # create the function for prediction/accuracy
    def predict(self, x_data):
        # load the model from storage
        with open('SVM_model.pkl', 'rb') as file2:
            lsvc_new = pickle.load(file2)
            return lsvc_new.predict(x_data)


# create an instance of the object
svm_object = SVM_facial_detection()
# svm_object.train_model()

# function for loading in a file and converting it into side
# new_photo = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Test-Images/Caleb_Test_1.jpg"
# new_photo = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Test-Images/Caleb_Test_2.jpg"
new_photo = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Test-Images/Caleb_Test_3.jpg"


def loadNewPhoto(new_photo):
    # change the photo to be an nparray
    opened_img = np.asarray(Image.open(new_photo).resize((500, 500)))
    # reshape
    plt.imshow(opened_img)
    plt.show()
    flattened_img = (opened_img.flatten()).reshape(1, -1)
    new_pred = svm_object.predict(flattened_img)
    # output the prediction
    print(f"The label for the input data was {new_pred}")


loadNewPhoto(new_photo=new_photo)