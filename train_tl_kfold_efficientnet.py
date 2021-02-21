import math, re, os, datetime
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
import leave_plot as lp
import leave_mixup as lm

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


def create_cnn_model():

    model = keras.models.Sequential()
    pre_trained_model = efn.EfficientNetB5(input_shape=(*IMG_SIZE, 3),
                                    include_top=False, 
                                    weights='noisy-student')

    # freeze the batch normalisation layers
    for layer in reversed(pre_trained_model.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.add(pre_trained_model)
    model.add(layers.Dropout(0.25))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.25))
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


def create_callbacks(model_name, log_dir):
    early_stopping = EarlyStopping(patience=3, monitor='val_loss', verbose=1)

    lr_schedule = LearningRateScheduler(lrfn, verbose=1)

    model_checkpoint = ModelCheckpoint(monitor='val_loss',
                                       filepath=model_name,
                                       save_best_only=True,
                                       verbose=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [
        early_stopping,
        lr_schedule,
        model_checkpoint,
        tensorboard_callback
    ]

    return callbacks


def train_model_naive_split():

    all_history = {}

    inp_train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=260,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
    )

    train_data = pd.read_csv('./data/merged_data.csv')
    train_data['label'] = train_data['label'].astype(str)
    Y = train_data[['label']]

    skf = KFold(n_splits = 5, random_state = 7, shuffle = True) 

    fold = 1
    for train_index, val_index in skf.split(np.zeros(len(train_data)),Y):

        training_data = train_data.iloc[train_index]
        validation_data = train_data.iloc[val_index]

        train_iterator1 = inp_train_gen.flow_from_dataframe(training_data,
                                                    x_col='image_id',
                                                    y_col='label',
                                                    directory='./data/train_images/',
                                                    target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    color_mode='rgb',
                                                    subset='training',
                                                    shuffle=True)

        train_iterator2 = inp_train_gen.flow_from_dataframe(training_data,
                                                            x_col='image_id',
                                                            y_col='label',
                                                            directory='./data/train_images/',
                                                            target_size=IMG_SIZE,
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical',
                                                            color_mode='rgb',
                                                            subset='training',
                                                            shuffle=True)

        # CutMixImageDataGenerator
        train_iterator = lm.CutMixImageDataGenerator(
            generator1=train_iterator1,
            generator2=train_iterator2,
            img_size=IMG_SIZE[0],
            batch_size=BATCH_SIZE,
        )

        validation_iterator = inp_train_gen.flow_from_dataframe(validation_data,
                                                    x_col='image_id',
                                                    y_col='label',
                                                    directory='./data/train_images',
                                                    target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=True)

        model = create_cnn_model()

        model_name = f'./output/models/best-model-kfold-{fold}.hdf5'

        log_dir = "./output/logs/fit/" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        history = model.fit(train_iterator,
                            steps_per_epoch=train_iterator.get_steps_per_epoch(),
                            validation_data=validation_iterator,
                            epochs=EPOCHS,
                            callbacks=create_callbacks(model_name, log_dir))

        all_history[f'history-fold-{fold}'] = history

        fold += 1

    return all_history

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


def load_and_predict_tta(model):

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

    model_files = os.listdir('./output/models/')
    print(model_files)
    models = []

    for model_file in model_files:
        print(f'loading model {model_file}')
        models.append(keras.models.load_model(f'./output/models/{model_file}', compile = True))

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

all_history = train_model_naive_split()
lp.plot_result('./output/graphs/', all_history)
store_prediction()