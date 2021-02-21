import pickle
import math, re, os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

import pandas as pd
import random
import shutil
import pathlib

BATCH_SIZE = 64
IMAGE_SIZE = [150, 150]
CLASSES = ['0', '1', '2', '3', '4']
EPOCHS = 30
AUTOTUNE = tf.data.experimental.AUTOTUNE

ALL_FILENAMES = []
for dirname, _, filenames in os.walk('./train_tfrecords'):
    for filename in filenames:
        ALL_FILENAMES.append(os.path.join(dirname, filename))
        print(os.path.join(dirname, filename))

TEST_FILENAMES = []
for dirname, _, filenames in os.walk('../test_tfrecords'):
    for filename in filenames:
        TEST_FILENAMES.append(os.path.join(dirname, filename))
        print(os.path.join(dirname, filename))
        
        
TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(
    ALL_FILENAMES,
    test_size=0.1, random_state=5
)


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def read_tf_record(example, labeled):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    if labeled:
        label = tf.cast(example['target'], tf.int64)
        label = tf.one_hot(label, 5)
        return image, label
    idnum = example['image_name']
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False, augment=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(partial(read_tf_record, labeled=labeled))
    if augment:
        dataset = dataset.map(data_augment)
    else: 
        dataset = dataset.map(data_only_resize)
    return dataset


def data_augment(image, label):
    # Thanks to the dataset.prefetch(AUTO) statement in the following function this happens essentially for free on TPU.
    # Data pipeline code is executed on the "CPU" part of the TPU while the TPU itself is computing gradients.
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, 0.1, 0.6)
#    image = tf.image.random_rotation(image, 360.0, fill_mode='reflect')
 #   image = tf.image.random_zoom(image)
 #   image = tf.image.random_shear(image)

    return image, label

def data_only_resize(image, label):
    # Thanks to the dataset.prefetch(AUTO) statement in the following function this happens essentially for free on TPU.
    # Data pipeline code is executed on the "CPU" part of the TPU while the TPU itself is computing gradients.
    image = tf.image.resize(image, IMAGE_SIZE)
    return image, label


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True, augment=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALID_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALID_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

print('Dataset: {} training images, {} validation images, {} (unlabeled) test images'.format(
    NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

def create_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=10e-5), metrics=['accuracy'])

    print(model.summary())
    return model


def create_callbacks():
    early_stopping = EarlyStopping(patience=6, monitor='val_loss', verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_lr=0.001,
                                  patience=6, mode='min',
                                  verbose=1)

    model_checkpoint = ModelCheckpoint(monitor='val_loss',
                                       filepath='./best-model.h5',
                                       save_best_only=True,
                                       verbose=1)

    callbacks = [
        early_stopping,
        reduce_lr,
        model_checkpoint
    ]

    return callbacks

def train_model_tf_records_naive_split():

    train_dataset = get_training_dataset()
    valid_dataset = get_validation_dataset()

    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
    VALID_STEPS = NUM_VALIDATION_IMAGES // BATCH_SIZE

    print(f'steps = {STEPS_PER_EPOCH}')
    print(f'valid steps= {VALID_STEPS}')

    model = create_cnn_model()

    history = model.fit(train_dataset,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALID_STEPS,
                        validation_data=valid_dataset,
                        epochs=EPOCHS,
                        callbacks=create_callbacks())

    return history

history = train_model_tf_records_naive_split()

def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.figure(figsize=(15, 5))
    plt.plot(epochs, acc, 'b*-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r*-', label='Validation accuracy')
    plt.grid()
    plt.title('Training and validation accuracy')
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.figure()
    plt.show()

    plt.figure(figsize=(15, 5))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b*-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
    plt.grid()
    plt.title('Training and validation loss')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.figure()
    plt.show()

plot_result(history)

def to_float32(image, label):
    return tf.cast(image, tf.float32), label

test_ds = get_test_dataset(ordered=True) 
test_ds = test_ds.map(to_float32)

print('Computing predictions...')
model = keras.models.load_model('./best-model.h5')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='image_id,label', comments='')
#head submission.csv

