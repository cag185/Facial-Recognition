# this file will create a SVM machine based on the training data provided from the user faces' databank
# SVM will be trained based on PCA dim reduced images

from sklearn.svm import LinearSVC
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

# create a class for the SVM model


class generic_SVM_model:
    def __init__(self):
        # import the data as an array
        parent_folder = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Facial-Profile-Databank/"
        face_list = os.listdir(parent_folder)
        # import array -- holds the flattened images
        flat_data_arr = []
        # label array -- holds the label of the images
        label_arr = []

    def loadData(self):
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
                label_arr.append(person_label)
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

    def trainModel(self):
        # split the data into testing and training data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=77, stratify=y)

        # create the SVM classifier
        lsvc = svm.LinearSVC(verbose=1, dual=auto)
        print(lsvc)

        # fit the model to the data
        lsvc.fit(x_train, y_train)

        print("The model has been successfully trained")

    def prediction(self):
        # test the model using the testing data
        y_pred = lsvc.predict(x_test)

    def getAccuracy(self):
        # calculate the accuracy of the model
        # compare the actual vs the prediction
        accuracy = accuracy_score(y_pred, y_test)
        print(f"The model accuracy is {accuracy * 100}% accurate")

    def runModel(self):
        self.loadData()
        self.trainModel()
        self.prediction()
        self.getAccuracy()


# create a new instance of the model class
model1 = generic_SVM_model()
model1.runModel()
