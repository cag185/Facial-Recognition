# Facial-Recognition
This repositiory will contain the code that does the facial detection software for our Senior Design project 

Requirements:
* Python
* Raspberry Pi
* Raspberry Pi Camera

Key Features:
* Software that takes a video feed from the Rasp Pi Camera and parcels it up into individual frames.
* Haarcascade library to limit photos to faces detected.
* PCA to extract important features from the frames and train the model for facial detection.
* Support Vector Machine learning model for the training.
* An additional class will be used in the one vs all approach to distinguish "non-viable" faces. This can include things like pictures of faces, ID cards, or screenshots of a face on a mobile phone.

Additonal Libraries:
* SciKit

Additional Resources:
* https://web.archive.org/web/20161010175545/https://tobilehman.com/blog/2013/01/20/extract-array-of-frames-from-mp4-using-python-opencv-bindings/
* https://www.enjoyalgorithms.com/blog/image-compression-using-pca
* https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# Copywrite:
The CelebA database is not for commercial or resale use but rather educational only. Here is the copywrite information:
 Large-scale CelebFaces Attributes (CelebA) Dataset
 By Multimedia Lab, The Chinese University of Hong Kong


For more information about the dataset, visit the project website:

  http://personal.ie.cuhk.edu.hk/~lz013/projects/CelebA.html

If you use the dataset in a publication, please cite the paper below:

  @inproceedings{liu2015faceattributes,
 	author = {Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang},
 	title = {Deep Learning Face Attributes in the Wild},
 	booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 	month = December,
 	year = {2015} 
  }
Please note that we do not own the copyrights to these images. Their use is RESTRICTED to non-commercial research and educational purposes.


## The following files need to be ran in order
* Convert_video_to_frames.py -- Converts all video files in the Video_to_split folder into frames
* FeatureExtraction.py -- uses haarcascade filter on images in the frames folder to crop to just faces
* PCA_compression.py -- compresses all the filtered images into much smaller images that are easier to train the SVM on
* SVM_model.py -- creates the SVM based on the PCA image files in the databank folders. The file is saved as a deseralized object. Can be used in other programs at this point
* ToggleLock.py -- Eventually will be used to unlock/alert the users of unathorized tampering. Currently takes in sample test photos and outputs the result

# To setup a virtual enviornment
* navigate to the root directory of the repo
* in linux enviornment run the command 'python -m venv {name_of_environment}'
* Activate the enviornment (in Linux) by running the command 'source {name_of_enviornment}/bin/activate'
* from here you can get all the requirements by running the command 'pip install -r requirements.txt' if the requirements page exists.
