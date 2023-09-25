# this program should perform PCA on the images in the image DataSet so that
# they are ready to be used in SVM
# from PIL import Image, ImageOps
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os

# function for running the PCA
def perform_pca(image):
    # convert the image into a numpy array
    img = np.array(image.getdata())
    img = img.reshape(*image.size, -1) # reshape into 3D array

    pca_channel = {}
    img_trans = np.transpose(img)

    for i in range(img.shape[-1]): # for each RGB channel do the PCA
        rgb_channel = img_t[i]
        channel = rgb_channel.reshape(*img.shape[:-1])
        pca = PCA(random_state= 45)
        fit_pca = pca.fit_transform(channel)
        pca_channel[i] = (pca, fit_pca) # saves PCA models for each channel

    return pca_channel # returns the dictionary of models

# include the different files in the training image data folder

img_source_path = ("C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/frames")

images = os.listdir(img_source_path)
for img in images:
    # perform the PCA
    pca_channel = perform_pca(img);