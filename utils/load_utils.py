import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from tqdm.auto import tqdm

def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])

def pad_img(img, pad_size=3, size=(28, 28)):
    return (cv2.resize(
        cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=255), 
        size
    ))

def erode_img(img, kernel=np.ones((2,2),np.uint8), iters=2, size=(28, 28)):
    return (cv2.resize(
        cv2.erode(img, kernel=kernel, iterations=iters), 
        size
    ))

def load_digi():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_all = np.concatenate((X_train, X_test))
    y_all = np.concatenate((y_train,  y_test))
    del X_train, y_train, X_test, y_test
    X_all = 255 - X_all
    X_all = X_all.astype('float32')
    X_all /= 255.0
    X_all = X_all.reshape(-1, 28, 28, 1)
    y_all = to_categorical(y_all)
    
    return X_all, y_all

def load_alpha():
    train = pd.read_csv('../data/emnist/emnist-balanced-train.csv', header=None)
    test = pd.read_csv('../data/emnist/emnist-balanced-test.csv', header=None)
    X_train = train.iloc[:, 1:]
    y_train = train.iloc[:, 0]
    X_test = test.iloc[:, 1:]
    y_test = test.iloc[:, 0]
    del train, test
    X_train = X_train[y_train >= 10].values
    y_train = pd.get_dummies(y_train[y_train >= 10] - 10).values
    X_test = X_test[y_test >= 10].values
    y_test = pd.get_dummies(y_test[y_test >= 10] - 10).values
    X_all = np.concatenate((X_train, X_test))
    y_all = np.concatenate((y_train,  y_test))
    del X_train, y_train, X_test, y_test
    X_all = np.apply_along_axis(rotate, 1, X_all)
    X_all = 255 - X_all
    X_all = X_all.astype('float32')
    X_all /= 255.0
    X_all = X_all.reshape(-1, 28, 28, 1)
    
    return X_all, y_all

def load_alnum():
    train = pd.read_csv('../Data/emnist/emnist-balanced-train.csv', header=None)
    test = pd.read_csv('../Data/emnist/emnist-balanced-test.csv', header=None)
    X_train = train.iloc[:, 1:].values
    y_train = pd.get_dummies(train.iloc[:, 0]).values
    X_test = test.iloc[:, 1:].values
    y_test = pd.get_dummies(test.iloc[:, 0]).values
    del train, test
    X_all = np.concatenate((X_train, X_test))
    y_all = np.concatenate((y_train,  y_test))
    del X_train, y_train, X_test, y_test
    X_all = np.apply_along_axis(rotate, 1, X_all)
    X_all = 255 - X_all
    X_all = X_all.astype('float32')
    X_all /= 255.0
    X_all = X_all.reshape(-1, 28, 28, 1)
    
    return X_all, y_all
    
def load_kdigi():
    train = pd.read_csv('../Data/kdigi/train.csv')
    test = pd.read_csv('../Data/kdigi/test.csv')   
    X_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values
    X_test = test.iloc[:, 1:].values
    y_test = test.iloc[:, 0].values
    del train, test
    X_all = np.concatenate((X_train, X_test))
    y_all = np.concatenate((y_train,  y_test))
    del X_train, y_train, X_test, y_test
    X_all = 255 - X_all
    X_all = X_all.astype('float32')
    X_all /= 255.0
    X_all = X_all.reshape(-1, 28, 28, 1)
    y_all = pd.get_dummies(y_all).values
    
    return X_all, y_all

def load_ddigi():
    X = []
    y = []
    for digit in tqdm(os.listdir('../data/ddigi/numerals-augmented/')):
        for img in os.listdir(f'../data/ddigi/numerals-augmented/{digit}'):
            i = cv2.cvtColor(plt.imread(f'../data/ddigi/numerals-augmented/{digit}/{img}', 0), cv2.COLOR_RGB2GRAY)
            X.append(i)
            y.append(digit) 
    X = np.array(X)
    y = np.array(y)
    p = np.random.permutation(len(X))
    X, y =  X[p], y[p]  
    X_cpy = X
    X = []
    for img in X_cpy:
        X.append(pad_img(img, 4))
    X = np.array(X)
    X = X.astype('float32')
    X /= 255.0
    X = X.reshape(-1, 28, 28, 1)
    y = pd.get_dummies(y).values
    
    return X, y

def load_maths():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_all = np.concatenate((X_train, X_test))
    y_all = np.concatenate((y_train,  y_test))
    del X_train, X_test, y_train, y_test  
    X_all = 255 - X_all
    X = []
    y = []
    for i, label in tqdm(enumerate(os.listdir('../Data/maths-balanced/'))):
        for img in os.listdir(f'../Data/maths-balanced/{label}'):
            im = cv2.cvtColor(plt.imread(f'../Data/maths-balanced/{label}/{img}', 0), cv2.COLOR_RGB2GRAY)
            X.append(im)
            y.append(i+10)
            
    X = np.array(X)
    y = np.array(y)
    p = np.random.permutation(len(X))
    X, y =  X[p], y[p] 
    
    X_cpy = X
    X = []
    for img in X_cpy:
        X.append(pad_img(img, pad_size=5, size=(55, 55)))
    X = np.array(X)
    del X_cpy
    
    X_cpy = X
    X = []
    for img in X_cpy:
        X.append(erode_img(img))
    X = np.array(X)
    del X_cpy
    print(X.shape, y.shape, X_all.shape, y_all.shape)
    
    X_all = np.concatenate((X_all, X))
    y_all = np.concatenate((y_all,  y))
    del X, y
    
    X_all = X_all.astype('float32')
    X_all /= 255.0
    X_all = X_all.reshape(-1, 28, 28, 1)
    y_all = to_categorical(y_all)
    
    return X_all, y_all