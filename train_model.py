# import json
import os

from keras import Input

# from distutils.dir_util import copy_tree
# import shutil
from utils import get_filename_only
import pandas as pd

# TensorFlow and tf.keras
import tensorflow as tf

# from tensorflow.keras import backend as K
print('TensorFlow version: ', tf.__version__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=4000)])

# Set to force CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

dataset_path = '.\\split_dataset\\'

tmp_debug_path = '.\\tmp_debug'
print('Creating Directory: ' + tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)
from keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0
# from keras.applications import MobileNetV2
# from keras.models import Sequential
import tensorflow.keras as keras
from keras.layers import Dense, Dropout, CuDNNGRU, ConvLSTM2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

input_size = 128
batch_size_num = 48
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_datagen = ImageDataGenerator(
    rescale=1 / 255,  # rescale the tensor values to [0,1]
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",  # "categorical", "binary", "sparse", "input"
    batch_size=batch_size_num,
    shuffle=True
    # save_to_dir = tmp_debug_path
)

val_datagen = ImageDataGenerator(
    rescale=1 / 255  # rescale the tensor values to [0,1]
)

val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",  # "categorical", "binary", "sparse", "input"
    batch_size=batch_size_num,
    shuffle=True
    # save_to_dir = tmp_debug_path
)

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

# Train a CNN classifier
# mobileNet = MobileNetV2(
#     input_shape=(input_size, input_size, 3),
#     alpha=1.4,
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     pooling='max',
#     classes=2,
#     classifier_activation="softmax",
# )

# efficient_net = EfficientNetB0(
#     weights='imagenet',
#     input_shape=(input_size, input_size, 3),
#     include_top=False,
#     pooling='max'
# )

input_layer = Input(shape=(input_size, input_size, 3))
efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(input_size, input_size, 3),
    include_top=False,
    pooling='max'
)(input_layer)
# efficient_net_output = efficient_net.get_layer('max_pool').output
eno_ext = tf.expand_dims(efficient_net, 0)
# x = (CuDNNGRU(units=1280))(eno_ext)
# x = (CuDNNGRU(units=256))(x)
x = (Dense(units=512, activation='relu'))(efficient_net)
x = (Dropout(0.7))(x)
x = (Dense(units=128, activation='relu'))(x)
# x = (Dropout(0.5))(x)

output = (Dense(units=1, activation='sigmoid'))(x)

model = keras.Model(inputs=input_layer, outputs=output)
# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_filepath = '.\\tmp_checkpoint'
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

# callbacks
custom_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath, 'best_model.h5'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
]

# Train network
num_epochs = 20
history = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=custom_callbacks
)
print(history.history)

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

# Plot results
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.plot(test_results['Prediction'], 'bo', label='Training loss')
plt.legend()
plt.show()
