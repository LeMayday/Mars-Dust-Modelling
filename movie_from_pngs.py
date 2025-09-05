# Author: Chris May
# 3/23/2025

import cv2
import os

def delete_files(filenames):
    # delete temp png files
    for file in set(filenames):
        if os.path.exists(file):
            os.remove(file)

def create_movie(filenames, output_mp4_file):
    # write to mp4
    # https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
    frame = cv2.imread(filenames[0])
    height, width, layers = frame.shape # type: ignore
    fourcc = cv2.VideoWriter.fourcc(*'mp4v') # mp4 codec
    frame_duration = 0.1
    video = cv2.VideoWriter(output_mp4_file, fourcc, 1/frame_duration, (width,height))
    for image in filenames:
        video.write(cv2.imread(image)) # type: ignore
    cv2.destroyAllWindows()
    video.release()
