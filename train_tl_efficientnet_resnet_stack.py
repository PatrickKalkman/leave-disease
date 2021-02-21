import pickle
import math, re, os, datetime
from os import path
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision
import numpy as np
from tensorflow.keras.models import Model
import efficientnet.tfkeras as efn

# Seed value
# Apparently you may use different seed values at each stage
seed_value = 12234

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)

import pandas as pd
import shutil
import pathlib

import leave_data as ld
import leave_plot as lp
import leave_mixup as lm
import cosine_lr as clr

# mixed_precision.set_global_policy('mixed_float16')

EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = (512, 512)

# BASE_FOLDER = '/kaggle/input/cassava-leaf-disease-classification/'
# WORKING_FOLDER = '/kaggle/working/'

BASE_FOLDER = './data/'
WORKING_FOLDER = './'

CSV_LOCATION = f'{BASE_FOLDER}merged_data.csv'
TRAINING_IMAGES_INPUT = f'{BASE_FOLDER}train_images/'
TEST_IMAGES_INPUT = f'{BASE_FOLDER}test_images/'
SUBMISSION_FILE = f'{WORKING_FOLDER}submission.csv'


def create_combined_model():
    pre_trained_efn = efn.EfficientNetB4(input_shape=(*IMG_SIZE, 3),
                                         include_top=False,
                                         weights='noisy-student')

    # freeze the batch normalisation layers
    for layer in reversed(pre_trained_efn.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    x = pre_trained_efn.output
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    prediction1 = layers.Dense(5, activation='softmax')(x)

    model_efn = Model(inputs=pre_trained_efn.input, outputs=prediction1)

    pre_trained_resnet = tf.keras.applications.ResNet50V2(input_shape=(*IMG_SIZE, 3),
                                                          include_top=False,
                                                          weights='imagenet')

    # freeze the batch normalisation layers
    for layer in reversed(pre_trained_resnet.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    x = pre_trained_resnet.output
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    prediction2 = layers.Dense(5, activation='softmax')(x)

    model_res = Model(inputs=pre_trained_resnet.input, outputs=prediction2)

    merged = layers.Concatenate([model_efn.output, model_res.output])

    merged = layers.Flatten()(merged)
    merged = layers.Dropout(0.5)(merged)
    merged = layers.Dense(1024, activation='relu')(merged)
    merged = layers.Dense(5, activation='softmax')(merged)

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    model_fusion = Model([model_efn.input, model_res.input], merged)
    model_fusion.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    print(model_fusion.summary())
    return model_fusion


def create_callbacks(log_dir):
    # early_stopping = EarlyStopping(patience=4, monitor='val_loss', verbose=1)

    lr_schedule = LearningRateScheduler(clr.lrfn, verbose=1)

    model_name = f'./output/models/best-model.hdf5'

    model_checkpoint = ModelCheckpoint(monitor='val_loss',
                                       filepath=model_name,
                                       save_best_only=True,
                                       verbose=1,
                                       pooling='average')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [
        lr_schedule,
        model_checkpoint,
        tensorboard_callback,
    ]

    return callbacks


def train_model_naive_split():
    inp_train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=120,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.2,
        zoom_range=0.4,
        brightness_range=[0.2, 1.0],
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        fill_mode='reflect',
    )

    train_data = pd.read_csv('./data/merged_data.csv')

    # 9500
    to_remove = np.random.choice(train_data[train_data['label'] == 3].index, size=9500, replace=False)
    train_data = train_data.drop(to_remove)

    train_data['label'] = train_data['label'].astype(str)
    Y = train_data[['label']]

    train_iterator = inp_train_gen.flow_from_dataframe(train_data,
                                                       x_col='image_id',
                                                       y_col='label',
                                                       directory='./data/train_images/',
                                                       target_size=IMG_SIZE,
                                                       batch_size=BATCH_SIZE,
                                                       class_mode='categorical',
                                                       subset='training',
                                                       shuffle=True)

    validation_iterator = inp_train_gen.flow_from_dataframe(train_data,
                                                            x_col='image_id',
                                                            y_col='label',
                                                            directory='./data/train_images/',
                                                            target_size=IMG_SIZE,
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical',
                                                            subset='validation',
                                                            shuffle=True)

    model = create_combined_model()

    log_dir = "./output/logs/fit/" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    history = model.fit(train_iterator,
                        validation_data=validation_iterator,
                        epochs=EPOCHS,
                        callbacks=create_callbacks(log_dir))

    return history


def train_model_naive_split_mixup():
    inp_train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=260,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        validation_split=0.25
    )

    train_data = pd.read_csv('./data/merged_data.csv')
    train_data['label'] = train_data['label'].astype(str)
    Y = train_data[['label']]

    train_iterator = lm.MixupImageDataGenerator(
        generator=inp_train_gen,
        directory='./data/train_images/',
        img_width=IMG_SIZE[0],
        img_height=IMG_SIZE[1],
        batch_size=BATCH_SIZE,
        subset='training'
    )

    validation_iterator = inp_train_gen.flow_from_dataframe(train_data,
                                                            x_col='image_id',
                                                            y_col='label',
                                                            directory='./data/train_images/',
                                                            target_size=IMG_SIZE,
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical',
                                                            color_mode='rgb',
                                                            subset='validation',
                                                            shuffle=True)

    model = create_cnn_model()

    log_dir = "./output/logs/fit/" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    history = model.fit(train_iterator,
                        validation_data=validation_iterator,
                        epochs=EPOCHS,
                        callbacks=create_callbacks(log_dir))

    return history


def load_and_predict(model):
    test_generator = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=360,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='nearest')

    ids = []
    tta_predictions = []

    for i in tqdm(range(10)):
        test_iterator = test_generator.flow_from_directory(
            './test/',
            target_size=IMG_SIZE,
            shuffle=False,
            class_mode='categorical',
            batch_size=1)

        if i == 1:
            for filename in test_iterator.filenames:
                print(filename)
                ids.append(filename.split('/')[1])

        predict_result = model.predict(test_iterator, steps=len(test_iterator.filenames))
        tta_predictions.append(predict_result)

    result = []
    predictions = np.mean(tta_predictions, axis=0)
    for index, prediction in enumerate(predictions):
        classes = np.argmax(prediction)
        result.append([ids[index], classes])
    result.sort()

    return result


def store_prediction():
    model = keras.models.load_model('./output/models/best-model.hdf5', compile=True)

    pathlib.Path(f'./test/1/').mkdir(parents=True, exist_ok=True)

    test_images = os.listdir(TEST_IMAGES_INPUT)
    ld.copy_test_images(test_images, TEST_IMAGES_INPUT)

    predictions = load_and_predict(model)

    # clean temp files
    if os.path.exists("./train"):
        shutil.rmtree('./train')

    if os.path.exists("./test"):
        shutil.rmtree('./test')

    df = pd.DataFrame(data=predictions, columns=['image_id', 'label'])
    df = df.set_index(['image_id'])

    if os.path.exists(SUBMISSION_FILE):
        os.remove(SUBMISSION_FILE)

    print(df.head())
    print('Writing submission')
    df.to_csv(SUBMISSION_FILE)


history = train_model_naive_split()
all_history = []
all_history.append(history)
lp.plot_result('./output/graphs', all_history)
store_prediction()
