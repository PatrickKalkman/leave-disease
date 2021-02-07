import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop

EPOCHS = 50
BATCH_SIZE = 32

#BASE_FOLDER = '/kaggle/input/cassava-leaf-disease-classification/'
#WORKING_FOLDER = '/kaggle/working/'

BASE_FOLDER = './data/'
WORKING_FOLDER = './'

CSV_LOCATION = f'{BASE_FOLDER}train.csv'
TRAINING_IMAGES_INPUT = f'{BASE_FOLDER}train_images/'
TEST_IMAGES_INPUT = f'{BASE_FOLDER}test_images/'
SUBMISSION_FILE = f'{WORKING_FOLDER}submission.csv'


def create_cnn_model():
    local_weights_file = './local_weights.h5'

    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=None)

    pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.4)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(5, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])

    print(model.summary())
    return model


def create_callbacks():
    early_stopping = EarlyStopping(patience=8, monitor='val_loss', verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_lr=0.001,
                                  patience=8, mode='min',
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
        rotation_range=270,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.1, 0.9],
        channel_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    train_iterator = train_gen.flow_from_directory('./train/train',
                                                   target_size=(150, 150),
                                                   batch_size=BATCH_SIZE,
                                                   class_mode='categorical')

    validation_gen = ImageDataGenerator(rescale=1. / 255.0)
    validation_iterator = validation_gen.flow_from_directory('./train/validate',
                                                             target_size=(150, 150),
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
    copy_test_images(test_images)

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

distribute_images(0.95)
history = train_model_naive_split()
plot_result(history)
store_prediction()





