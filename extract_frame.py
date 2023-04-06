import os
import cv2
import math
from utils import get_filename_only

videos_path = '.\\train_sample_videos\\'
frames_path = '.\\train_sample_frames\\'

print('Creating Directory: ' + frames_path)
os.makedirs(frames_path, exist_ok=True)

videos = []
for filename in os.listdir(videos_path):
    if not os.path.isdir(os.path.join(videos_path, filename)):
        continue

    print('Creating Directory: ' + os.path.join(frames_path, filename))
    os.makedirs(os.path.join(frames_path, filename), exist_ok=True)

    tmp_path = os.path.join(videos_path, get_filename_only(filename))
    print('Processing Directory: ' + tmp_path)
    for folder in os.listdir(tmp_path):
        videos = [x for x in os.listdir(tmp_path)
                  if os.path.isfile(os.path.join(tmp_path, x))]

    for video in videos:
        print(os.path.join(tmp_path, video))
        # tmp_path = os.path.join(frames_path, filename)
        # print('Creating Directory: ' + tmp_path)

        print('Converting Video to Images...')
        count = 0
        video_file = os.path.join(tmp_path, video)
        cap = cv2.VideoCapture(video_file)
        # frame rate
        frame_rate = cap.get(5)

        while cap.isOpened():
            frame_id = cap.get(20)  # current frame number
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

                new_filename = '{}-{:03d}.png'.format(os.path.join(frames_path, filename, get_filename_only(video)), count)
                # new_filename = '{}-{:03d}.png'.format(tmp_path, count)
                count = count + 1
                cv2.imwrite(new_filename, new_frame)
                if count == 5:
                    break
        cap.release()
        print("Done!")
    else:
        continue
