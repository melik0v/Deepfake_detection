import sys
import os
import cv2
import math
from utils import get_filename_only

videos_path = '.\\train_sample_videos\\'
frames_path = '.\\train_sample_frames\\'

"""
Extract frame sequence (10 frames) from video
"""

with open(os.path.join(videos_path, 'List_of_testing_videos.txt')) as file:
    metadata = []
    for line in file:
        if sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
            metadata.append(line.split()[1])
        elif sys.platform == 'win32':
            metadata.append(line.split()[1].replace('/', '\\'))

# for filename in os.listdir(videos_path):

for filename in metadata:
    if filename.endswith(".mp4"):
        print(filename)
        tmp_path = os.path.join(frames_path, filename)
        print('Creating Directory: ' + tmp_path)
        os.makedirs(tmp_path, exist_ok=True)
        print('Converting Video to Images...')
        count = 0
        video_file = os.path.join(videos_path, filename)
        cap = cv2.VideoCapture(video_file)
        # frame rate
        frame_rate = cap.get(5)

        while cap.isOpened():
            frame_id = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % math.floor(frame_rate) == 0:
                print('Original Dimensions: ', frame.shape)
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif 1000 < frame.shape[1] <= 1900:
                    scale_ratio = 0.5
                else:
                    scale_ratio = 1
                print('Scale Ratio: ', scale_ratio)

                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dim = (width, height)
                new_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                print('Resized Dimensions: ', new_frame.shape)

                new_filename = '{}-{:03d}.png'.format(os.path.join(tmp_path, get_filename_only(filename)), count)
                # new_filename = '{}-{:03d}.png'.format(tmp_path, count)
                count = count + 1
                cv2.imwrite(new_filename, new_frame)
        cap.release()
        print("Done!")
    else:
        continue
