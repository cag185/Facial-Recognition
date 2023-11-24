# this file will create a SVM machine based on the training data provided from the user faces' databank
# SVM will be trained based on PCA dim reduced images

from sklearn.svm import LinearSVC
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import cv2


class SVM_facial_detection():
    # import the data as an array
    training_acc = 0

    def __init__(self):
        self.training_acc = None

    def train_model(self):
        parent_folder = "Facial-Profile-Databank/"
        face_list = os.listdir(parent_folder)
        # import array -- holds the flattened images
        flat_data_arr = []
        # label array -- holds the label of the images
        label_arr = []

        # iterating var
        person_label = 0
        # per each person that has a facial profile
        for person in face_list:
            print("loading....person: " + person)
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
            print("loaded person: " + person + " successfully")
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

        # get a very small sample of the data to train the classifier params on
        # since this bullshit isnt workin just hardcode with c=10
        # subset_size = int(len(y_train) / 5)
        # subset_indices = np.random.choice(
        #     (x_train.shape[0]), size=subset_size, replace=False)
        # subset_indices = subset_indices.astype(int)
        # x_train_subset = []
        # y_train_subset = []
        # try:
        #     for num in subset_indices:
        #         x_train_subset.append(x_train[num])
        #         y_train_subset.append(y_train[num])
        # except KeyError as e:
        #     print("KeyError:", e)
        # # parameters
        # param_grid = {
        #     'C': [0.01, 0.1, 1, 10, 100],
        # }

        # standardize the data
        scaler = StandardScaler()

        # xtrain and x_test
        x_train_fit = scaler.fit_transform(x_train)
        x_test_fit = scaler.fit_transform(x_test)

        # create the SVM classifier
        print("Creating the svm.LinearSVC")
        svc = svm.LinearSVC(dual=True, max_iter=1000, C=10, random_state=42)
        print("Done creating the svm.LinearSVC")

        # create the randomized searchCV object
        # random_search = RandomizedSearchCV(
        #     svc, param_distributions=param_grid, n_iter=10, cv=5, n_jobs=1, random_state=42
        # )

        # random_search.fit(x_train_subset, y_train_subset)

        # find the best model
        # best_model = random_search.best_estimator_
        # test_score = svc.score(x_test_fit, y_test)
        # print("Best Params: ", random_search.best_params_)
        # print(
        #     "Best Cross-validated Accuracy: {:.2f}".format(random_search.best_score_))
        # print("Test Accuracy with best params: {:.2f}".format(test_score))

        # fit the model to the data
        # lsvc.fit(x_train, y_train)
        # lsvc.fit(x_train_fit, y_train)
        # print("The classifier has been trained on the small sample dataset")

        print("retraining here")
        best_model = svc.fit(x_train_fit, y_train)
        print("retraining succesfull")
        # we save the best model
        self.best_model = best_model
        # self.lsvc = lsvc

        # # test the model using the testing data
        # this is for the whole data brought in
        # y_pred = (lsvc.predict(x_test))
        y_pred = (best_model.predict(x_test_fit))
        # compare the actual vs the prediction
        # this is for the whole data brought in
        training_acc = accuracy_score(y_pred, y_test)
        print(
            "The training accuracy for the data in the training mode: " + str(training_acc))
        # save the training accuracy
        self.training_acc = training_acc

        # once we have created the model, we want to save it
        with open('SVM_model_larger.pkl', 'wb') as file:
            pickle.dump(best_model, file)

    # create the function for prediction/accuracy
    def predict(self, x_data):
        # load the model from storage
        with open('SVM_model.pkl', 'rb') as file2:
            lsvc_new = pickle.load(file2)
            return lsvc_new.predict(x_data)


# create an instance of the object
svm_object = SVM_facial_detection()

# train the model
# svm_object.train_model()

# once the model is trained can retrieve the accuracy
# print("training accuracy:" + svm_object.training_acc + "accurate")

# function for loading in a file and converting it into side


def loadNewPhoto(new_photo):
    # change the photo to be cv2 in split colors
    img = cv2.cvtColor(cv2.imread(new_photo), cv2.COLOR_BGR2RGB)
    full_image = cv2.resize(img, (500, 500))
    # split rgb
    blue, green, red = cv2.split(full_image)
    norm_blue = blue/255
    norm_green = green/255
    norm_red = red/255

    pca_b = PCA(n_components=.99)
    pca_b.fit(norm_blue)
    trans_pca_b = pca_b.transform(norm_blue)

    pca_g = PCA(n_components=.99)
    pca_g.fit(norm_green)
    trans_pca_g = pca_g.transform(norm_green)

    pca_r = PCA(n_components=.99)
    pca_r.fit(norm_red)
    trans_pca_r = pca_r.transform(norm_red)

    # recombine the images
    b_arr = pca_b.inverse_transform(trans_pca_b)
    g_arr = pca_g.inverse_transform(trans_pca_g)
    r_arr = pca_r.inverse_transform(trans_pca_r)

    # merge the photos back to one
    img_reduced = (cv2.merge((r_arr, g_arr, b_arr)))

    # change the photo to be an nparray
    # opened_img = np.asarray(Image.open(new_photo).resize((500, 500)))
    # reshape
    # plt.imshow(opened_img)
    # plt.show()
    flattened_img = (img_reduced.flatten()).reshape(1, -1)

    # need to also compress the image using PCA

    new_pred = svm_object.predict(flattened_img)
    # output the prediction
    print("The label for the input data was " + new_pred)


# loadNewPhoto(new_photo=new_photo)

# load in the testing images for Caleb
# caleb_dir = "Facial-Recognition/Test-Images/Caleb"
caleb_dir = "Test-Images/Caleb"
caleb_files = os.listdir(caleb_dir)
print("------ Testing images for Caleb ------")
for img in caleb_files:
    full_file = caleb_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Cam
# cam_dir = "Facial-Recognition/Test-Images/Cam"
cam_dir = "Test-Images/Cam"
cam_files = os.listdir(cam_dir)
print("------ Testing images for Cam ------")
for img in cam_files:
    full_file = cam_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Hudson
# hudson_dir = "Facial-Recognition/Test-Images/Hudson"
hudson_dir = "Test-Images/Hudson"
hudson_files = os.listdir(hudson_dir)
print("------ Testing images for Hudson ------")
for img in hudson_files:
    full_file = hudson_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for Lucas
# lucas_dir = "Facial-Recognition/Test-Images/Lucas"
lucas_dir = "Test-Images/Lucas"
lucas_files = os.listdir(lucas_dir)
print("------ Testing images for Lucas ------")
for img in lucas_files:
    full_file = lucas_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for MATT
# matt_dir = "Facial-Recognition/Test-Images/Matt"
matt_dir = "Test-Images/Matt"
matt_files = os.listdir(matt_dir)
print("------ Testing images for Matt ------")
for img in matt_files:
    full_file = matt_dir + "/" + img
    loadNewPhoto(full_file)

# load in the testing images for JACK
# jack_dir = "Facial-Recognition/Test-Images/Jack"
jack_dir = "Test-Images/Jack"
jack_files = os.listdir(jack_dir)
print("------ Testing images for Jack ------")
for img in jack_files:
    full_file = jack_dir + "/" + img
    loadNewPhoto(full_file)
