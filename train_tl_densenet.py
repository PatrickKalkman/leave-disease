import pickle
import math, re, os
from os import path

import tensorflow as tf
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201

import pandas as pd
import random
import shutil
import pathlib

import leave_data as ld
import leave_plot as lp

EPOCHS = 50
BATCH_SIZE = 32
IMG_SIZE = (150, 150)

#BASE_FOLDER = '/kaggle/input/cassava-leaf-disease-classification/'
#WORKING_FOLDER = '/kaggle/working/'

BASE_FOLDER = './data/'
WORKING_FOLDER = './'

CSV_LOCATION = f'{BASE_FOLDER}train.csv'
TRAINING_IMAGES_INPUT = f'{BASE_FOLDER}train_images/'
TEST_IMAGES_INPUT = f'{BASE_FOLDER}test_images/'
SUBMISSION_FILE = f'{WORKING_FOLDER}submission.csv'


def create_cnn_model():
    model = keras.models.Sequential()
    pre_trained_model = DenseNet201(input_shape=(*IMG_SIZE, 3),
                                    weights='imagenet',
                                    include_top=False)

    model.add(pre_trained_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=1e-5), metrics=['accuracy'])

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


def train_model_naive_split():

    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    train_iterator = train_gen.flow_from_directory('./train/train',
                                                   target_size=IMG_SIZE,
                                                   batch_size=BATCH_SIZE,
                                                   class_mode='categorical')

    validation_gen = ImageDataGenerator(rescale=1. / 255.0)
    validation_iterator = validation_gen.flow_from_directory('./train/validate',
                                                             target_size=IMG_SIZE,
                                                             batch_size=BATCH_SIZE,
                                                             class_mode='categorical')

    model = create_cnn_model()

    history = model.fit(train_iterator,
                        validation_data=validation_iterator,
                        epochs=EPOCHS,
                        callbacks=create_callbacks())

    return history



def load_and_predict(model):

    test_generator = ImageDataGenerator(rescale=1. / 255)

    test_iterator = test_generator.flow_from_directory(
        './test/',
        target_size=(150, 150),
        shuffle=False,
        class_mode='categorical',
        batch_size=1) 

    ids = []
    for filename in test_iterator.filenames:
        print(filename)
        ids.append(filename.split('/')[1])

    predict_result = model.predict(test_iterator, steps=len(test_iterator.filenames))
    predictions = []
    for index, prediction in enumerate(predict_result):
        classes = np.argmax(prediction)
        predictions.append([ids[index], classes])
    predictions.sort()

    return predictions


def store_prediction():
    model = keras.models.load_model('./best-model.h5', compile = True)

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

ld.distribute_images(0.9, CSV_LOCATION, TRAINING_IMAGES_INPUT)
history = train_model_naive_split()
lp.plot_result(history)
store_prediction()