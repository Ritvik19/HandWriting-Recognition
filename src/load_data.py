import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

import os
import sys
sys.path.append('../')

from utils.load_utils import *

def load_image_sets(dataset_id, image_size):
    datagen = ImageDataGenerator(
        rescale = 1./255, 
        rotation_range=10, 
        zoom_range = 0.1,
        width_shift_range=0.1, 
        height_shift_range=0.1,
        validation_split=0.1
    )
    
    color_mode = 'grayscale' if image_size[2] == 1 else 'rgb'
    class_mode = 'binary' if len(os.listdir(f'../data/{dataset_id}/')) == 2 else 'categorical'
    
    training_set = datagen.flow_from_directory(
        f'../data/{dataset_id}/', target_size=image_size[:2],  batch_size=32, class_mode=class_mode, 
        subset='training', color_mode=color_mode
    )
    validation_set = datagen.flow_from_directory(
        f'../data/{dataset_id}/', target_size=image_size[:2],  batch_size=32, class_mode=class_mode, 
        subset='validation', color_mode=color_mode
    )
    print(training_set.class_indices)
    return training_set, validation_set

def create_image_sets(dataset_id, image_size):
    if dataset_id == 'digi':
        X_all, y_all = load_digi()
    if dataset_id == 'alpha':
        X_all, y_all = load_alpha()
    if dataset_id == 'alnum':
        X_all, y_all = load_alnum()
    if dataset_id == 'kdigi':
        X_all, y_all = load_kdigi()
    if dataset_id == 'ddigi':
        X_all, y_all = load_ddigi()
    if dataset_id == 'maths':
        X_all, y_all = load_maths()   

    datagen = ImageDataGenerator(
        rotation_range=10, 
        zoom_range = 0.1,
        width_shift_range=0.1, 
        height_shift_range=0.1,
        validation_split=0.1
    )  
    datagen.fit(X_all)
    training_set = datagen.flow(
        x=X_all, y=y_all,
        subset='training'
    )
    validation_set = datagen.flow(
        x=X_all, y=y_all,
        subset='validation'
    )
    
    return training_set, validation_set
