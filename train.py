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
print("Loaded train_a data. Took: {:.3f} seconds".format(timeit.default_timer()-start))
start = timeit.default_timer()
train_b = np.array([np.asarray(Image.open(img_path)) for img_path in train_b_dir])
print("Loaded train_b data. Took: {:.3f} seconds".format(timeit.default_timer()-start))


