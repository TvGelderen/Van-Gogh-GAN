import tensorflow as tf
import numpy as np
import pathlib
from PIL import Image


data_dir = 'data/'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# 'A' contains van Gogh paintings, 'B' contains pictures
train_a_dir, train_b_dir = list(data_dir.glob('train_a/*')), list(data_dir.glob('train_b/*'))
test_a_dir, test_b_dir = list(data_dir.glob('test_a/*')), list(data_dir.glob('test_b/*'))
# Turn the images into numpy arrays
train_a = np.array([np.asarray(Image.open(img_path)) for img_path in train_a_dir])
train_b = np.array([np.asarray(Image.open(img_path)) for img_path in train_b_dir])
