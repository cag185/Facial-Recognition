# https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118
# this might be helpful in updating the effeciency of PCA

import numpy as np
from sklearn.decomposition import PCA
from PIL import Image, ImageOps
import matplotlib.pyplot as plt 
import sys # used to get size data
import cv2
import math
import os

# TODO- for each image in the video capture do the PCA dimensionality reduction
# load in the image
# parent_img_folder = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/frames/Caleb_face_2/"
parent_img_folder = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/frames/Hudson_face/"

# parent_img_folder = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/frames/FlowerFrame/"
img_folder_arr = os.listdir(parent_img_folder)
frame_count = 0
# imgPath = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/frames/FaceFrames/frame145.jpg"
# imgPath = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/frames/FlowerFrame/flower.jpg"
for i in img_folder_arr:
    imgPath = parent_img_folder + i
    # tab from here down
    # imgPath = parent_img_folder + 
    img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    # full_image = Image.open(img_I).resize((500,500))
    full_image = cv2.resize(img,(500,500))

    # split the image into RGB values
    blue, green, red  = cv2.split(full_image)
    # divide the data by 255 to normalize the data (0-1)
    norm_blue = blue/255
    norm_green = green/255
    norm_red = red/255

    # fit each image using PCA where n_components is the variance we want to achieve
    n_var = .99
    pca_b = PCA(n_components=n_var)
    pca_b.fit(norm_blue)
    trans_pca_b = pca_b.transform(norm_blue)

    pca_g = PCA(n_components=n_var)
    pca_g.fit(norm_green)
    trans_pca_g = pca_g.transform(norm_green)

    pca_r = PCA(n_components=n_var)
    pca_r.fit_transform(norm_red)
    trans_pca_r = pca_r.transform(norm_red)


    # print off the contained variance by the three channels
    # print(f"Blue Channel: {sum(pca_b.explained_variance_ratio_)}")
    # print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
    # print(f"Red Channel: {sum(pca_r.explained_variance_ratio_)}")

    # recombine the images
    b_arr = pca_b.inverse_transform(trans_pca_b)
    g_arr = pca_g.inverse_transform(trans_pca_g)
    r_arr = pca_r.inverse_transform(trans_pca_r)

    # merge the photos back to one
    img_reduced = (cv2.merge((r_arr, g_arr, b_arr)))
    img_reduced_for_display =(cv2.merge((b_arr, g_arr, r_arr )))
    img_to_write = np.clip((img_reduced / img_reduced.max()) * 255, 0, 255).astype(np.uint8)

    # DONT NEED TO PLOT AND SHOW EACH IMAGE
    # fig = plt.figure(figsize = (10, 7.2))
    # fig.add_subplot(121)
    # plt.title("Original Image")
    # plt.imshow(full_image)

    # fig.add_subplot(122)
    # plt.title("Reduced Image")
    # plt.imshow(img_reduced_for_display)

    # plt.show()

    # save the file
    fileDest = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Facial-Profile-Databank/Hudson_1/face_%d.jpg" % frame_count
    # fileDest = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Facial-Profile-Databank/Caleb_2/face_%d.jpg" % frame_count
    cv2.imwrite(fileDest, img_reduced_for_display)
    # Increase the frame counter
    frame_count+=1

# # NOT NEEDED RN
# # Get the file size
    size_reconstruct = os.path.getsize(fileDest) / 1024
    print("The size of the reconstructed file (kb): ", size_reconstruct)
    print("The size of the original file (kb): ", os.path.getsize(imgPath) / 1024)
    # get the number of components
    totalNumComp = pca_b.explained_variance_ratio_.shape[0] + pca_g.explained_variance_ratio_.shape[0] + pca_r.explained_variance_ratio_.shape[0]
    print("The number of components used to achieve ", n_var, "variance : ", totalNumComp, ".")
