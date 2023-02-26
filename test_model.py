from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from efficientnet.tfkeras import EfficientNetB0
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import os

pd.set_option('display.max_rows', 500)
dataset_path = '.\\split_dataset\\'

input_size = 128
batch_size_num = 32
train_path = os.path.join(dataset_path, 'train')
train_path_1 = os.path.join(dataset_path, 'train_1')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')
test_path_1 = os.path.join(dataset_path, 'test_1')


checkpoint_filepath = '.\\tmp_checkpoint'

test_datagen = ImageDataGenerator(
    rescale=1 / 255  # rescale the tensor values to [0,1]
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    classes=['real', 'fake'],
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode=None,
    batch_size=1,
    shuffle=False
)

# load the saved model that is considered the best
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))

# Generate predictions
test_generator.reset()

preds = best_model.predict(
    test_generator,
    verbose=1
)

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})
print(test_results)

import matplotlib.pyplot as plt

plt.plot(test_results['Prediction'], 'bo', label='Training loss')
plt.legend()
plt.show()
