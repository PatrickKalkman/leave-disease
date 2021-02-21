import pickle
import math, re, os
from os import path
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import efficientnet.tfkeras as efn
from tensorflow.keras import mixed_precision

import pandas as pd
import shutil
import pathlib

import leave_data as ld
import leave_plot as lp

EPOCHS = 50
BATCH_SIZE = 10
IMG_SIZE = (512, 512)

#BASE_FOLDER = '/kaggle/input/cassava-leaf-disease-classification/'
#WORKING_FOLDER = '/kaggle/working/'

BASE_FOLDER = './data/'
WORKING_FOLDER = './'

CSV_LOCATION = f'{BASE_FOLDER}merged_data.csv'
TRAINING_IMAGES_INPUT = f'{BASE_FOLDER}train_images/'
TEST_IMAGES_INPUT = f'{BASE_FOLDER}test_images/'
SUBMISSION_FILE = f'{WORKING_FOLDER}submission.csv'

mixed_precision.set_global_policy('mixed_float16')

def create_cnn_model():

    model = keras.models.Sequential()
    pre_trained_model = efn.EfficientNetB4(input_shape=(*IMG_SIZE, 3),
                                    include_top=False, 
                                    weights='noisy-student')

    # freeze the batch normalisation layers
    for layer in reversed(pre_trained_model.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.add(pre_trained_model)
    model.add(layers.Dropout(0.4))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(5, activation='softmax'))

    # add metrics
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    ]

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(model.summary())
    return model


LEARNING_RATE = 3e-5
LR_START = 1e-8
LR_MIN = 1e-8
LR_MAX = LEARNING_RATE
LR_RAMPUP_EPOCHS = 3
LR_SUSTAIN_EPOCHS = 0
N_CYCLES = .5


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        progress = (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) / (EPOCHS - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)
        lr = LR_MAX * (0.5 * (1.0 + tf.math.cos(math.pi * N_CYCLES * 2.0 * progress)))
        if LR_MIN is not None:
            lr = tf.math.maximum(LR_MIN, lr)
            
    return lr


def create_callbacks():
    early_stopping = EarlyStopping(patience=3, monitor='val_loss', verbose=1)

    lr_schedule = LearningRateScheduler(lrfn, verbose=1)

    model_checkpoint = ModelCheckpoint(monitor='val_loss',
                                       filepath='./best-model-efn.h5',
                                       save_best_only=True,
                                       verbose=1)

    callbacks = [
        early_stopping,
        lr_schedule,
        model_checkpoint,
    ]

    return callbacks


def train_model_naive_split():

    inp_train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=260,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Create training and validation generator.
    # train_iterator = lm.MixupImageDataGenerator(generator=inp_train_gen,
    #                                           directory='./train/train',
    #                                           batch_size=BATCH_SIZE,
    #                                           img_height=IMG_SIZE[0],
    #                                           img_width=IMG_SIZE[1],
    #                                           subset='training')

    train_iterator = inp_train_gen.flow_from_directory('./train/train',
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
                        #steps_per_epoch=train_iterator.get_steps_per_epoch(),
                        validation_data=validation_iterator,
                        epochs=EPOCHS,
                        callbacks=create_callbacks())

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