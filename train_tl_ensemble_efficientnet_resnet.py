import os
from os import path
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import efficientnet.tfkeras as efn
from sklearn.model_selection import StratifiedKFold, KFold

import pandas as pd
import shutil
import pathlib

import leave_data as ld


EPOCHS = 30
BATCH_SIZE = 8
IMG_SIZE = (512, 512)

#BASE_FOLDER = '/kaggle/input/cassava-leaf-disease-classification/'
#WORKING_FOLDER = '/kaggle/working/'

BASE_FOLDER = './data/'
WORKING_FOLDER = './'

CSV_LOCATION = f'{BASE_FOLDER}merged_data.csv'
TRAINING_IMAGES_INPUT = f'{BASE_FOLDER}train_images/'
TEST_IMAGES_INPUT = f'{BASE_FOLDER}test_images/'
SUBMISSION_FILE = f'{WORKING_FOLDER}submission.csv'

def load_and_predict(models):

    test_generator = ImageDataGenerator(rescale=1. / 255)

    model_predictions = []

    model_index = 1

    for model in models:
        print(f'predicting model {model_index}')
        test_iterator = test_generator.flow_from_directory(
            './test/',
            target_size=IMG_SIZE,
            shuffle=False,
            class_mode='categorical',
            batch_size=1) 

        ids = []
        for filename in test_iterator.filenames:
            print(filename)
            ids.append(filename.split('/')[1])

        predict_result = model.predict(test_iterator, steps=len(test_iterator.filenames))
        model_predictions.append(predict_result)
        model_index += 1

    result = []
    predictions = np.mean(model_predictions, axis=0)
    for index, prediction in enumerate(predictions):
        classes = np.argmax(prediction)
        result.append([ids[index], classes])
    result.sort()

    return result


def store_prediction():

    model_files = os.listdir('./output/models/')
    print(model_files)
    models = []

    models.append(keras.models.load_model(f'./output/models/best-model-efnet.hdf5', compile = True))
    models.append(keras.models.load_model(f'./output/models/best-model-resnet.hdf5', compile = True))

    pathlib.Path(f'./test/1/').mkdir(parents=True, exist_ok=True)
    test_images = os.listdir(TEST_IMAGES_INPUT)
    ld.copy_test_images(test_images, TEST_IMAGES_INPUT)
    predictions = load_and_predict(models)

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

store_prediction()