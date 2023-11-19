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
* https://www.enjoyalgorithms.com/blog/image-compression-using-pca

## The following files need to be ran in order
* Convert_video_to_frames.py -- Converts all video files in the Video_to_split folder into frames
* FeatureExtraction.py -- uses haarcascade filter on images in the frames folder to crop to just faces
* PCA_compression.py -- compresses all the filtered images into much smaller images that are easier to train the SVM on
* SVM_model.py -- creates the SVM based on the PCA image files in the databank folders. The file is saved as a deseralized object. Can be used in other programs at this point
* ToggleLock.py -- Eventually will be used to unlock/alert the users of unathorized tampering. Currently takes in sample test photos and outputs the result
  
