# this file will test the trained model against a database of faces
import cv2
import numpy as np
import os
import pickle
from PIL import Image

# import data to work with
unauthorized_face_list = []
unauthorized = 0
authorized = 1
unauthorized_comparator = []
unauthorized_dir = "Video-to-frames/haar_cascade_frames/unauthorized_users/"
authorized_dir = "Video-to-frames/haar_cascade_frames/"

# load in the svc
with open('OCSVM_model.pkl', 'rb') as file:
    lsvc = pickle.load(file)

foo_count = 0
error_count = 0
for foo in os.listdir(unauthorized_dir):
    if foo_count > 1000:
        break
    opened_img = np.asarray(Image.open(
        unauthorized_dir + foo).convert('L').resize((100, 100)))
    new_img = (opened_img.flatten()).reshape(1, -1)
    pred = lsvc.predict(new_img)
    unauthorized_face_list.append(pred)
    unauthorized_comparator.append(unauthorized)

    # count the difference
    if pred != unauthorized:
        error_count += 1
    foo_count += 1
# get the accuracy
total_acc = (len(unauthorized_face_list) - error_count) / \
    len(unauthorized_face_list)

# print this
print(
    f"The total accuracy of correctly classifying as not-authorized: {total_acc}")

# get the accuracy for the images in the other folders

# error tracker
total_authorized_size = 0
error_size = 0
for person in os.listdir(authorized_dir):
    new_dir = authorized_dir + person
    # dont use the non-authorized ones
    if (person == "unauthorized_users"):
        break
    for image in os.listdir(new_dir):
        # load the image and convert to the proper form for use in model
        img_path = new_dir
        loadedImg = np.asarray(Image.open(new_dir + "/"
                                          + image).convert('L').resize((100, 100)))
        new_img1 = (opened_img.flatten()).reshape(1, -1)
        # test the image in the classifier
        predict = lsvc.predict(new_img1)
        if (predict != authorized):
            error_size += 1
        total_authorized_size += 1

total_acc_auth = (total_authorized_size - error_size) / \
    total_authorized_size

# print this
print(
    f"The total accuracy of correctly classifying as authorized: {total_acc_auth}")
