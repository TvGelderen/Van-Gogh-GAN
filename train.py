import tensorflow as tf
import numpy as np
import pathlib
from PIL import Image
import timeit

# Randomize the random seed
np.random.seed(seed=None)

BATCH_SIZE = 10
EPOCHS = 50

data_dir = 'data/'
data_dir = pathlib.Path(data_dir)
# 'A' contains van Gogh paintings, 'B' contains pictures
real_images_dir = list(data_dir.glob('*'))
# Turn the images into numpy arrays
start = timeit.default_timer()
real_images = np.array([np.asarray(Image.open(img_path)) for img_path in real_images_dir])
print("Loaded the images, elapsed time: {:.3f} seconds".format(timeit.default_timer()-start))


# Discriminator input (256, 256, 3) and output (2) with softmax
#   conv2d layers eventually flattened to a dense layer of 1 neuron
def create_discriminator():
    discriminator_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=24, kernel_size=(6, 6), activation='relu', input_shape=[256, 256, 3]),
        tf.keras.layers.Conv2D(filters=18, kernel_size=(6, 6), activation='relu'),
        tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return discriminator_model


# Generator input (240, 240, 1) and output (256, 256, 3)
#   transposed conv2d (or deconvolutional) layers
def create_generator():
    generator_model = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(filters=36, kernel_size=(5, 5), activation='relu', use_bias=False, input_shape=[240, 240, 1]),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(filters=30, kernel_size=(5, 5), activation='relu', use_bias=False),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(filters=24, kernel_size=(5, 5), activation='relu', use_bias=False),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), activation='relu', use_bias=False),
    ])
    return generator_model


discriminator = create_discriminator()
discriminator.summary()
generator = create_generator()
generator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Returns a generated noise matrix of 240 by 240, with values from -1 to 1
def generate_noise():
    return np.random.uniform(-1.0, 1.0, (240, 240))


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


@tf.function
def train_step(images):
    noise_array = np.array([generate_noise() for i in range(BATCH_SIZE)])

    with tf.GradientTape as gen_tape, tf.GradientTape() as dis_tape:
        generated_images = generator(noise_array)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        dis_loss = discriminator_loss(real_output, fake_output)

    grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    grad_dis = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(grad_dis, discriminator.trainable_variables))


def train():
    for epoch in EPOCHS:
        start_time = timeit.default_timer()

        for iteration in range(len(real_images.shape[0])/BATCH_SIZE):
            index = np.array([np.random.randint(0, 399) for i in range(BATCH_SIZE)])
            images = np.array([real_images[idx] for idx in index])
            train_step(images)

        print("Epoch {}/{}\t\tTime elapsed".format(epoch, EPOCHS, (timeit.default_timer()-start_time)))

    generator.save('model/generator')
    discriminator.save('model/discriminator')


def generate_images():
    for i in range(5):
        img_array = np.squeeze(generator.predict(np.expand_dims(generate_noise(), axis=0)), axis=0)
        img = Image.fromarray(img_array, 'RGB')
        img.save('output/{:03d}.jpg'.format(i))
        img.show()


train()

# generated_output = np.squeeze(generator.predict(np.expand_dims(real_images[0], axis=0)), axis=0)
#
# img = Image.fromarray(generated_output, 'RGB')
# img.save('output/0001.jpg')
# img.show()
