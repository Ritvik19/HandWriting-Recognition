from keras.models import Sequential
from keras.applications import MobileNetV2
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

def cnn(image_size, output_classes):
    classifier = Sequential()
    if output_classes == 10:
        classifier.add(Conv2D(32, (5, 5), input_shape=image_size, activation='relu', padding='same'))
        classifier.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    else:
        classifier.add(Conv2D(128, (5, 5), input_shape=image_size, activation='relu', padding='same'))
        classifier.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))
    if output_classes == 10:
        classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    else:
        classifier.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        classifier.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))
    classifier.add(Flatten())
    if output_classes == 2:
        classifier.add(Dense(units=1, activation='softmax'))
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    else:
        classifier.add(Dense(units=output_classes, activation='softmax'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return classifier

models = {
    'lens-digi': cnn,
    'lens-alpha': cnn,
    'lens-alnum': cnn,
    'lens-kdigi': cnn,
    'lens-ddigi': cnn,
    'lens-maths': cnn,
}

