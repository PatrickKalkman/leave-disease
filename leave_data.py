import random
import shutil
import pathlib
from os import path
import pandas as pd

def distribute_images(train_validation_split_ratio, csv_location, input_path):
    create_folder_structure()
    train_df = pd.read_csv(csv_location)

    for label in train_df['label'].unique():
        print(f'processing: {label}')
        pathlib.Path(f'./train/train/{label}').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'./train/validate/{label}').mkdir(parents=True, exist_ok=True)

        labels_df = train_df[train_df.label.eq(label)]
        images = labels_df['image_id'].tolist()
        random.shuffle(images)
        split_index = int(len(images) * train_validation_split_ratio)

        training_images = images[:split_index]
        validation_images = images[split_index:]

        copy_images(training_images, f'train/{label}', input_path)
        copy_images(validation_images, f'validate/{label}', input_path)


def create_folder_structure():
    if path.exists('./train'):
        shutil.rmtree('./train')
    pathlib.Path("./train/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./train/validate").mkdir(parents=True, exist_ok=True)


def copy_images(source_list, destination_path, input_path):
    for image in source_list:
        shutil.copyfile(f'{input_path}{image}', f'./train/{destination_path}/{image}')


def copy_test_images(source_list, input_path):
    for image in source_list:
        shutil.copyfile(f'{input_path}{image}', f'./test/1/{image}')