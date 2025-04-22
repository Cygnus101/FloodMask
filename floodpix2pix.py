import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import * 
from tensorflow.keras.activations import * 
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import os
from tqdm import tqdm
import numpy as np
import random
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanIoU
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from glob import glob

def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + smooth) / (union + smooth)

data = np.load('/kaggle/input/floodmapping2/kaggle/working/dataset/input/input_0.npy')

# Print the shape (dimensions)
print("Shape of input_0.npy:", data.shape)


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=not apply_batchnorm))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
    result.add(layers.Conv2D(filters, size, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def Generator():
    inputs = layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (128x128x64)
        downsample(128, 4),                        # (64x64x128)
        downsample(256, 4),                        # (32x32x256)
        downsample(512, 4),                        # (16x16x512)
        downsample(512, 4),                        # (8x8x512)
        downsample(512, 4),                        # (4x4x512)
        downsample(512, 4),                        # (2x2x512)
        downsample(512, 4),                        # (1x1x512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),      # (2x2x1024)
        upsample(512, 4, apply_dropout=True),      # (4x4x1024)
        upsample(512, 4, apply_dropout=True),      # (8x8x1024)
        upsample(512, 4),                          # (16x16x1024)
        upsample(256, 4),                          # (32x32x512)
        upsample(128, 4),                          # (64x64x256)
        upsample(64, 4),                           # (128x128x128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(1, 4, strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='sigmoid')  # instead of sigmoid

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')
    tar = layers.Input(shape=[256, 256, 1], name='target_image')

    x = layers.concatenate([inp, tar])  # (256, 256, 3)

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                         use_bias=False)(zero_pad1)
    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

def generator_loss(disc_generated_output, gen_output, target):
    # GAN loss (using Binary Crossentropy)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # Dice loss (for segmentation quality)
    dice_loss_value = dice_loss(target, gen_output)
    
    return gan_loss + (100 * dice_loss_value)

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))


input_files = sorted(glob('/kaggle/input/floodmapping2/kaggle/working/dataset/input/*.npy'))
target_files = sorted(glob('/kaggle/input/floodmapping2/kaggle/working/dataset/target/*.npy'))

def load_npy_pair(input_path, target_path):
    # Load the input and target .npy files
    input_array = np.load(input_path.decode()).astype(np.float32)  # shape: (256, 256, 2)
    target_array = np.load(target_path.decode()).astype(np.float32)  # shape: (256, 256, 1)
    
    # Calculate 3rd channel = input[..., 0] - input[..., 1]
    diff_channel = np.expand_dims(input_array[..., 0] - input_array[..., 1], axis=-1)  # shape: (256, 256, 1)

    # Concatenate to form (256, 256, 3)
    input_array = np.concatenate([input_array, diff_channel], axis=-1)

    return input_array, target_array

def tf_load_npy(input_path, target_path):
    input_img, target_img = tf.numpy_function(
        load_npy_pair, [input_path, target_path], [tf.float32, tf.float32])
    input_img.set_shape([256, 256, 3])  # Now 3 channels
    target_img.set_shape([256, 256, 1])
    return input_img, target_img

def augment_image(input_img, target_img):
    # Random flip left-right
    flip_lr = tf.random.uniform(()) > 0.5
    input_img = tf.cond(flip_lr, lambda: tf.image.flip_left_right(input_img), lambda: input_img)
    target_img = tf.cond(flip_lr, lambda: tf.image.flip_left_right(target_img), lambda: target_img)

    # Random flip up-down
    flip_ud = tf.random.uniform(()) > 0.5
    input_img = tf.cond(flip_ud, lambda: tf.image.flip_up_down(input_img), lambda: input_img)
    target_img = tf.cond(flip_ud, lambda: tf.image.flip_up_down(target_img), lambda: target_img)

    # Random rotation (0, 90, 180, 270 degrees)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    input_img = tf.image.rot90(input_img, k)
    target_img = tf.image.rot90(target_img, k)

    return input_img, target_img


def make_dataset(input_paths, target_paths, batch_size=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((input_paths, target_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(input_paths))
    dataset = dataset.map(tf_load_npy, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentations
    dataset = dataset.map(lambda x, y: augment_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = make_dataset(input_files, target_files)

def generate_images(model, input_image, target, step):
    prediction = model(input_image, training=False)
    
    plt.figure(figsize=(15, 5))

    display_list = [
        input_image[0, :, :, 0],  # Channel 1
        input_image[0, :, :, 1],  # Channel 2
        input_image[0, :, :, 2],  # Difference (Channel 3)
        tf.squeeze(target[0], axis=-1), 
        tf.squeeze(prediction[0], axis=-1)
    ]
    
    title = ['Input (ch 1)', 'Input (ch 2)', 'Diff (ch 3)', 'Target', 'Prediction']

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"sample_step_{step}.png")
    plt.show()

EPOCHS = 20

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for step, (input_image, target) in enumerate(train_ds):
        train_step(input_image, target)

        # Show image after every 100 steps or first batch
        if step % 100 == 0 or step == 0:
            generate_images(generator, input_image, target, step=epoch*100 + step)