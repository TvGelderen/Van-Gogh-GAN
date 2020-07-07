import tensorflow as tf
import numpy as np
import pathlib
from PIL import Image
import timeit

data_dir = 'data/'
data_dir = pathlib.Path(data_dir)
# 'A' contains van Gogh paintings, 'B' contains pictures
train_a_dir, train_b_dir = list(data_dir.glob('train_a/*')), list(data_dir.glob('train_b/*'))
test_a_dir, test_b_dir = list(data_dir.glob('test_a/*')), list(data_dir.glob('test_b/*'))
# Turn the images into numpy arrays
start = timeit.default_timer()
train_a = np.array([np.asarray(Image.open(img_path)) for img_path in train_a_dir])
print("Loaded train_a data. Elapsed time: {:.3f} seconds".format(timeit.default_timer()-start))
start = timeit.default_timer()
# train_b = np.array([np.asarray(Image.open(img_path)) for img_path in train_b_dir])
print("Loaded train_b data. Elapsed time: {:.3f} seconds".format(timeit.default_timer()-start))

# Discriminator input (256, 256, 3) and output (2) with softmax
#   conv3d and conv2d layers eventually flattened to a dense layer of 2 neurons
# Generator input (256, 256, 3) and output (256, 256, 3)
#   transposed conv3d (or deconvolutional) layers

discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=24, kernel_size=(6, 6), activation='relu', input_shape=[256, 256, 3]),
    tf.keras.layers.Conv2D(filters=18, kernel_size=(6, 6), activation='relu'),
    tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

discriminator.build()
discriminator.summary()

generator = tf.keras.Sequential([
    tf.keras.layers.Conv2DTranspose(filters=36, kernel_size=(5, 5), activation='relu', input_shape=[256, 256, 3]),
    tf.keras.layers.Conv2DTranspose(filters=30, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.Conv2DTranspose(filters=24, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.Cropping2D(cropping=((8, 8), (8, 8))),
    tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), activation='relu'),
])

generator.build()
generator.summary()
