import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

es = EarlyStopping(monitor='loss', mode='min', verbose=1)
filepath = "model.h5"
ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

image_size = (28, 28)

datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = False,
                                   validation_split=0.3)
                                   
training_set = datagen.flow_from_directory('../[Data] HandWriting-Recognition/MathSymbols',
                                                 target_size = image_size,
                                                 color_mode='grayscale',
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 subset='training')

validation_set = datagen.flow_from_directory('../[Data] HandWriting-Recognition/MathSymbols',
                                                 target_size = image_size,
                                                 color_mode='grayscale',
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 subset='validation')

def cnn(image_size):
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (*image_size, 1), activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 17, activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.summary()
    return classifier

# neuralnetwork = cnn(image_size)

neuralnetwork = load_model("model.h5")

neuralnetwork.fit_generator(training_set,
                         steps_per_epoch = 139612,
                         epochs = 50,
                         validation_data = validation_set,
                         validation_steps = 59825, 
                         callbacks=[es, ckpt])
