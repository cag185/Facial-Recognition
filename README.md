# Facial-Recognition
This repositiory will contain the code that does the facial detection software for our Senior Design project 

Requirements:
* Python
* Raspberry Pi
* Raspberry Pi Camera

Key Features:
* Software that takes a video feed from the Rasp Pi Camera and parcels it up into individual frames
* PCA to extract important features from the frames and train the model for facial detection
* Support Vector Machine learning model for the training
* An additional class will be used in the one vs all approach to distinguish "non-viable" faces. This can include things like pictures of faces, ID cards, or screenshots of a face on a mobile phone

Additonal Libraries:
* SciKit

Additional Resources:
* https://web.archive.org/web/20161010175545/https://tobilehman.com/blog/2013/01/20/extract-array-of-frames-from-mp4-using-python-opencv-bindings/
