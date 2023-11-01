# this file will create a SVM machine based on the training data provided from the user faces' databank
# SVM will be trained based on PCA dim reduced images

from sklearn.svm import LinearSVC
import os
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import pandas as pd

# import the data as an array
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
    # for each image in that profile
    image_dir = parent_folder + person + "/"
    img_list = os.listdir(image_dir)

    for image in img_list:
        full_path = image_dir + image

        # open the image as an array
        curr_image = Image.open(full_path)
        curr_image_nparray = np.array(curr_image)
        # flatten the image
        flat_data_arr.append(curr_image_nparray.flatten());
        # append it to a flattened array
        label_arr.append(person_label)
    
    # at the end, add 1 to the iterator
    person_label += 1

# at this point the two arrays from the start should be np arrays
label_data = np.array(label_arr)
print("done preprocessing")

# create a pandas dataframe out of the data
df = pd.DataFrame(flat_data_arr)
df['Label'] = label_data

# 1. Load the images and convert to data frame
# 2. Separate image into features and labels
# 3. Split test and training values
# 4. Build and train the SVM
# 5. Model evaulation (accuraccy)
# 6. Using model for prediction
#  


