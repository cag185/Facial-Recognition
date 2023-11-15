# reference https://web.archive.org/web/20161010175545/https://tobilehman.com/blog/2013/01/20/extract-array-of-frames-from-mp4-using-python-opencv-bindings/

import cv2
import os

# make the video bit more dynamic
parent_folder = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/video_to_split/"
sub_files = os.listdir(parent_folder)
# sub_files = ["Caleb_face.mp4", "Caleb_face_2.mp4", "Lucas_face.mp4", "Cam_face.mp4"]

# create paths
output_parent_directory = "C:/Users/17578/Desktop/School/Class Files/Fall 2023/ECE 1896 - Senior Design/Facial-Recognition/Video-to-frames/frames/"
# output_arr = ["caleb_1/", "caleb_2/", "lucas/", "cam/"]

file_count = 0
for subfile in sub_files:
    # open the video file
    full_file  = parent_folder + subfile
    vf = cv2.VideoCapture(full_file)

    # use the read method on the video capture object to return
    # a tuple where the first element is the success flag and the second is the
    # image array of rgb values
    success, image = vf.read()

    # for each of the above folder paths create the folders 
    # for output in output_arr:
    # make this more dynamic
    folder_name = output_parent_directory + subfile

    test_output = folder_name[:-4] + "/"
    # test_output = output_parent_directory + output_arr[file_count]
    doesExist = os.path.exists(test_output)
    if not doesExist:
        os.makedirs(test_output)
    frame_count = 0
    while success:
        success, image = vf.read()
        print(success)
        write_dir = test_output + "frame%d.jpg" % frame_count 
        if not (image is None):
            cv2.imwrite(write_dir, image)
            frame_count += 1
    
    # increase the file counter
    file_count += 1