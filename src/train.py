import numpy as np
import matplotlib.pyplot as plt
import load_data, models, model_dispatcher
from tqdm.auto import tqdm

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model

import argparse, os

        
def train_cnn(dataset_id, name, image_size=(28, 28, 1)):
    if dataset_id in ['digi', 'ddigi', 'kdigi']:
        output_classes = 10
    elif dataset_id == 'alpha':
        output_classes = 37
    elif dataset_id == 'alnum':
        output_classes = 47
    elif dataset_id == 'maths':
        output_classes = 17
    else:
        if os.path.exists(f'../data/{dataset_id}/training'):
            output_classes = len(os.listdir(f'../data/{dataset_id}/training'))
        else:
            output_classes = len(os.listdir(f'../data/{dataset_id}'))
        
    output_directory = os.path.join(os.path.dirname(os.getcwd()), 'performance', name)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)        
        
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=7)
    filepath = f'../models/{name}.h5'
    ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    rlp = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1)
    
    neuralnetwork = models.models[name](image_size, output_classes)
    
    plot_model(
        neuralnetwork, to_file=os.path.join(output_directory, 'arch.png'), 
        show_shapes=True, show_layer_names=True
    )
    
    training_set, validation_set = load_data.create_image_sets(dataset_id, image_size)
    
    history = neuralnetwork.fit_generator(
        training_set, validation_data=validation_set,
        callbacks=[es, ckpt, rlp], epochs=1000,
    )
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle('Model Performance', fontsize=24) 

    ax[0].plot(history.history['loss'], label='t-loss')
    ax[0].plot(history.history['val_loss'], label='v-loss')
    ax[0].set_title('Loss', fontsize=18)
    ax[0].set_ylabel('Loss')

    ax[1].plot(history.history['acc'], label='t-acc')
    ax[1].plot(history.history['val_acc'], label='v-acc')
    ax[1].set_title('Score', fontsize=18)
    ax[1].set_ylabel('Score')

    for i in range(2):
        ax[i].grid()
        ax[i].legend()
        ax[i].set_xlabel('Epochs')
        
    fig.savefig(os.path.join(output_directory, f'Model-Performance.png'))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    
    parser.add_argument('-a', '--algo', metavar='algo', type=str, help='algorithm to run')
    parser.add_argument('-d', '--dataset', metavar='data', type=str, help='dataset to train on')
    parser.add_argument('-p', '--problem', metavar='problem', nargs='?', type=str, help='type of problem')
    parser.add_argument('-t', '--type', metavar='type', nargs='?', type=str, help='type of dataset')
    
    args = parser.parse_args()
    print(args)
    train_cnn(args.dataset.strip(), args.algo.strip())