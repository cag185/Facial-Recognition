import numpy as np
from sklearn.decomposition import PCA
from PIL import Image, ImageOps
import matplotlib.pyplot as plt 
import sys # used to get size data

# load in the image
imgPath = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/frames/frame0.jpg"
full_image = Image.open(imgPath).resize((500,500))

print(full_image.format)
print(full_image.size)
print(full_image.mode)

# convert the image to a numpy array
img_arr = np.asarray(full_image)

# convert the 3 channel image to a single channel grey image
grayscale_arr = np.dot(img_arr[...,:3], [0.2989, 0.5870, 0.1140])

# flatten the image -- need an array of flattened images to pass into pca
# flattened_img = np.assimg_arr.flatten()
height, width = grayscale_arr.shape
flattened_img = grayscale_arr.reshape(-1,1)

# apply PCA
n_comp = 1 # change this depending on the variance needed
pca = PCA(n_components=n_comp)
compressed_img = pca.fit_transform(flattened_img)

# reconstruct the image
reconstructed_img = pca.inverse_transform(compressed_img)
reconstructed_img = reconstructed_img.reshape((height, width))

# show the OG, compressed, and reconstucted image
fig, ax = plt.subplots(1,3)

fig.suptitle("Image compression and reconstruction with N = " + str(n_comp) + " principle components")

subtitle_string = "Original Image \n Size of the image in kb: " + str(sys.getsizeof(img_arr) / 1000)
ax[0].imshow(img_arr.astype('uint8'))
ax[0].axis('off')
ax[0].set_title(subtitle_string, color='red')

subtitle_string = "Compressed Image \n Size of the image in kb: " + str(sys.getsizeof(compressed_img) / 1000)
ax[1].imshow(compressed_img.reshape((height, width, n_comp)).astype('uint8'))
ax[1].axis('off')
ax[1].set_title(subtitle_string, color='blue')


subtitle_string = "Reconstructed Image \n Size of the image in kb: " + str(sys.getsizeof(reconstructed_img) / 1000)
ax[2].imshow(reconstructed_img.astype('uint8'))
ax[2].axis('off')
ax[2].set_title(subtitle_string, color='green')

plt.tight_layout()
plt.show()
