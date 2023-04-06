import cv2
from mtcnn import MTCNN
import os.path
import tensorflow as tf
from utils import get_filename_only, rotate
from math import atan, pi

"""
Crop faces from frame sequence
"""

print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_path = '.\\train_sample_frames\\'
dataset_path = '.\\prepared_dataset\\'

real_path = os.path.join(dataset_path, 'real')
fake_path = os.path.join(dataset_path, 'fake')
os.makedirs(real_path, exist_ok=True)
os.makedirs(fake_path, exist_ok=True)

for filename in os.listdir(base_path):
    if not os.path.isdir(os.path.join(base_path, filename)):
        continue
    tmp_path = os.path.join(base_path, get_filename_only(filename))
    print('Processing Directory: ' + tmp_path)
    frame_images = [x for x in os.listdir(tmp_path)
                    if os.path.isfile(os.path.join(tmp_path, x))]
    print('Cropping Faces from Images...')

    for frame in frame_images:
        print('Processing ', frame)
        detector = MTCNN()
        image = cv2.cvtColor(cv2.imread(os.path.join(tmp_path, frame)), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        print('Face Detected: ', len(results))
        count = 0

        for result in results:
            bounding_box = result['box']
            print(bounding_box)
            confidence = result['confidence']
            print(confidence)
            if len(results) < 2 or confidence > 0.95:
                margin_x = bounding_box[2] * 0.5  # 30% as the margin
                margin_y = bounding_box[3] * 0.5  # 30% as the margin
                x1 = int(bounding_box[0] - margin_x)
                if x1 < 0:
                    x1 = 0
                x2 = int(bounding_box[0] + bounding_box[2] + margin_x)
                if x2 > image.shape[1]:
                    x2 = image.shape[1]
                y1 = int(bounding_box[1] - margin_y)
                if y1 < 0:
                    y1 = 0
                y2 = int(bounding_box[1] + bounding_box[3] + margin_y)
                if y2 > image.shape[0]:
                    y2 = image.shape[0]
                print(x1, y1, x2, y2)
                crop_image = image[y1:y2, x1:x2]
                # try:
                #     result = detector.detect_faces(crop_image)[0]
                # except IndexError:
                #     result = result

                keypoints = result['keypoints']
                # rotation
                alpha = atan(
                    (keypoints['left_eye'][1] - keypoints['right_eye'][1]) / (
                            keypoints['left_eye'][0] - keypoints['right_eye'][0])
                )
                crop_image = rotate(crop_image, alpha * 180 / pi)

                if 'real' in filename:
                    new_filename = '{}-{:02d}.png'.format(
                        os.path.join(dataset_path, 'real', get_filename_only(frame)), count
                    )
                else:
                    new_filename = '{}-{:02d}.png'.format(
                        os.path.join(dataset_path, 'fake', get_filename_only(frame)), count
                    )
                count = count + 1
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
            else:
                print('Skipped a face..')
