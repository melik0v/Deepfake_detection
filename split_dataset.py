import os.path
import numpy as np
import splitfolders

base_path = '.\\train_sample_videos\\'
dataset_path = '.\\prepared_dataset\\'

real_path = os.path.join(dataset_path, 'real')
fake_path = os.path.join(dataset_path, 'fake')

all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
print('Total Number of Real faces: ', len(all_real_faces))

all_fake_faces = [f for f in os.listdir(fake_path) if os.path.isfile(os.path.join(fake_path, f))]
print('Total Number of Fake faces: ', len(all_fake_faces))

random_faces = np.random.choice(all_fake_faces, len(all_real_faces), replace=False)


print('Down-sampling Done!')

# Split into Train/ Val/ Test folders
splitfolders.ratio(dataset_path, output='split_dataset', seed=1377, ratio=(0.8, 0.1, 0.1))  # default values
print('Train/ Val/ Test Split Done!')
