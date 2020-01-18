import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

X_train = pd.read_csv('../[Data] HandWriting-Recognition/emnist/emnist-balanced-train.csv', header=None).iloc[:, 1: ].values
X_test =  pd.read_csv('../[Data] HandWriting-Recognition/emnist/emnist-balanced-test.csv', header=None).iloc[:, 1:].values

y_train = pd.read_csv('../[Data] HandWriting-Recognition/emnist/emnist-balanced-train.csv', header=None)[0].values
y_test = pd.read_csv('../[Data] HandWriting-Recognition/emnist/emnist-balanced-test.csv', header=None)[0].values

print('Training Set')
print(len(y_train))

print('Testing Set')
print(len(y_test))

np.unique(y_train)

mapping = {}
with open('../[Data] HandWriting-Recognition/emnist/emnist-balanced-mapping.txt') as f:
    for line in f.readlines():
        a,b = line.strip().split()
        mapping[int(a)] = chr(int(b))
print(mapping)        

def sample(i):
    print(mapping[y_train[i]])
    plt.imshow(X_train[i].reshape(28,28),cmap=plt.get_cmap('gray'))

sample(0)
sample(1)


print("The EMNIST dataset is rotated and flipped and we need to fix that")

from keras.utils.np_utils import to_categorical

def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])
X_train = np.apply_along_axis(rotate, 1, X_train)/255
X_test = np.apply_along_axis(rotate, 1, X_test)/255

sample(0)
sample(1)

X_all = np.concatenate((X_train, X_test)).reshape(-1, 28, 28, 1)
y_all = np.concatenate((y_train,  y_test))
y_all = to_categorical(y_all)

del X_train, X_test, y_train, y_test

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='loss', mode='min', verbose=1)
filepath = "E:/Models/Alphabet-Digit-Recognition/CNN.h5"
ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

def cnn(image_size):
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (*image_size, 1), activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 47, activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.summary()
    return classifier

model = cnn((28,28))

model.fit(X_all, y_all, validation_split=0.3, epochs=500, callbacks=[es, ckpt])
