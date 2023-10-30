# this file will create a SVM machine based on the training data provided from the user faces' databank
# SVM will be trained based on PCA dim reduced images

from sklearn import svm, LinearSVC
import os
import numpy 
from PIL import Image

# import the data as an array
parent_folder = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Facial-Profile-Databank/user2/"
image_list = os.listdir(parent_folder)

for image in image_list:
    full_path = parent_folder + image
    # open the image as an array
    curr_image = Image.open(full_path)
    curr_image_nparray = asarray(curr_image)

    # now train the SVM model


