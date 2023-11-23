# This program will feed in test data (new persons face) and run that through the existing SVM model to detect a verified face
# WARNING--- for an image to be used in the SVM it must be loaded, converted into 500,500 image, and flattened
# reshape image to (1,-1) as one sample is used


# TODO: need to not allow the lock to toggle in the case where we are not seeing an authorized user as the classification

import pickle
import os
import numpy as np
from PIL import Image

# step 1 - load in the test image
unprocessed_img = "../Test-Images/Jack/Jack_Test_2.jpg"
opened_img = np.asarray(Image.open(unprocessed_img).resize((500, 500)))
new_image = (opened_img.flatten()).reshape(1, -1)

# step 2 - load in the SVM model
with open('../SVM_model.pkl', 'rb') as file:
    lsvc = pickle.load(file)

    # now want to pass in the data that we have processed
    prediction = lsvc.predict(new_image)

    print(f"Prediction.....{prediction}")
