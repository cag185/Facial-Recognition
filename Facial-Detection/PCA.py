# this program should perform PCA on the images in the image DataSet so that
# they are ready to be used in SVM
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# function for importing the image and its data
def image_data(imgPath, disp=True):
    original_img = Image.open(imgPath)

    img_size_kb = os.stat(imgPath).st_size/1024
    data  = original_img.getdata()
    original_pixels = np.array(data).reshape(*original_img.size, -1)
    img_dim = original_pixels.shape

    if disp:
        plt.imshow(original_img)
        plt.show()
    data_dict = {}
    data_dict['img_size_kb'] = img_size_kb
    data_dict['img_dim'] = img_dim

    return data_dict

# function for running the PCA
def perform_pca(image):
    # convert the image into a numpy array
    image = Image.open(image)
    img = np.array(image.getdata())
    img = img.reshape(*image.size, -1) # reshape into 3D array

    pca_channel = {}
    img_trans = np.transpose(img)

    for i in range(img.shape[-1]): # for each RGB channel do the PCA
        rgb_channel = img_trans[i]
        channel = rgb_channel.reshape(*img.shape[:-1])
        pca = PCA(random_state = 45)
        fit_pca = pca.fit_transform(channel)
        pca_channel[i] = (pca, fit_pca) # saves PCA models for each channel

    return pca_channel # returns the dictionary of models

 # function for reconstruction of the compressed image
 # n_components is the number of Principle components to save for reconstruction
def pca_transform(pca_channel, n_components):
    temp_res = []

    # loop over each channel
    for channel in range(len(pca_channel)):
        pca, fit_pca = pca_channel[channel]

        pca_pixel = fit_pca[:, :n_components]
        pca_computation = pca.components_[:n_components, :]
        compressed_pixel = np.dot(pca_pixel, pca_computation) + pca.mean_
        temp_res.append(compressed_pixel)

    #transform the channel width height to height width channel
    compressed_image = np.transpose(temp_res)
    # form compressed image
    compressed_image  = np.array(compressed_image, dtype=np.uint8)

    # finally return the compressed image
    return compressed_image


# include the different files in the training image data folder
img_source_path = ("C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/frames")
img_dest_path = ("C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Facial-Profile-Databank/user1/")

images = os.listdir(img_source_path)
img_count = 0
for img in images:
    #increment the counter
    img_count+=1

    # concat the image with the path
    newPath = img_source_path + "/" + img
    # get the image data
    data_dict = image_data(newPath, False)

    # perform the PCA
    print(newPath)
    pca_channel = perform_pca(newPath);
    img_res = pca_transform(pca_channel, 450)

    # write the image to a new file
    full_url = img_dest_path + "compressedImage%d.jpg" %img_count
    cv2.imwrite(full_url, img_res)

    print("End of the program")
