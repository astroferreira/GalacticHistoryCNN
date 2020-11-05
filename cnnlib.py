import numpy as np
import pandas as pd
import time
import tensorflow as tf
import keras
import math
import h5py
import os

from keras import regularizers
from keras.regularizers import *
from keras.datasets import mnist
from keras.models import Sequential

from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.legacy.layers import MaxoutDense

from keras.optimizers import *
from keras.callbacks import TensorBoard

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from keras.datasets import mnist
from keras.models import Sequential, Model

from keras.optimizers import *
from keras.callbacks import TensorBoard

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from keras.utils import plot_model
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from scipy.stats import zscore
from scipy.ndimage import rotate
import cnnlib as lib
from keras.initializers import glorot_uniform

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import keras.backend.tensorflow_backend as K


def metainfo(labels):

    train_size = labels.shape[0]
    number_of_classes = labels.shape[1]

    ids = np.linspace(0, train_size-1, train_size)
    total = labels.shape[0]
    class_size = np.zeros(number_of_classes)
    for i in range(number_of_classes):
        class_size[i] = labels.T[i].sum()
    
    max_class = np.max(class_size)
    print(class_size)
    cw = [max_class/class_size_i for class_size_i in class_size]
    print(cw)

    return ids, train_size, cw


def categorize(values, binsize, min, max):

    num_bins = int((max-min) // binsize)
    print(num_bins)
    labels = np.zeros((values.shape[0], num_bins))
    
    for i in range(num_bins):
        
        bin_min = min + i*binsize
        bin_max = min + (1+i)*binsize
        
        labels.T[i][np.where((values >= bin_min) & (values < bin_max))] = 1

    return labels

def undersample2(labels, ratio=1):
    
    
    nm = labels.T[0].sum()
    mm = labels.T[1].sum()
    
    target_normals = nm * ratio
    target_normals = int(np.ceil(target_normals))
  
    
    nm_n = np.where(labels.T[0] == 1)
    mm_n = np.where(labels.T[1] == 1)
    #nm = np.where(labels.T[2] == 1)
    
    print(nm_n)
    print(mm_n)
    #print(nm)
    
    if(target_normals < mm):
        new_mm = np.array(sorted(np.random.choice(mm_n[0], target_normals, replace=False)))
        print(new_mm.shape, nm_n)
    else:
        new_mm = mm[0]
    
    new_idx = sorted(np.hstack([nm_n[0], new_mm]))
    
    print([nc.sum() for nc in labels.T])
    print([nc.sum() for nc in labels[new_idx].T])
    
    return new_idx

def undersample(labels, ratio=1):
    
    
    nmergers = labels.T[0].sum() + labels.T[1].sum()
    n = labels.T[2].sum()
    
    target_normals = nmergers * ratio
    target_normals = int(np.ceil(target_normals))
  
    
    bm = np.where(labels.T[0] == 1)
    pm = np.where(labels.T[1] == 1)
    nm = np.where(labels.T[2] == 1)
    
    print(bm)
    print(pm)
    print(nm)
    
    if(target_normals < n):
        new_nm = sorted(np.random.choice(nm[0], target_normals, replace=False))
    else:
        new_nm = nm[0]
    
    new_idx = sorted(np.hstack([bm[0], pm[0], new_nm]))
    
    print([nc.sum() for nc in labels.T])
    print([nc.sum() for nc in labels[new_idx].T])
    
    return new_idx

def norm_morpho(morpho):
    
    morpho[np.isnan(morpho)] = 0
    
    for i in range(morpho.shape[1]):
        
        morpho.T[i] -= morpho.T[i].mean()
        morpho.T[i] /= morpho.T[i].std()
        print(morpho.T[i].mean(), morpho.T[i].std())

    morpho[np.isnan(morpho)] = 0
        
    return morpho


def process_data(x):
    x = x.astype('float32')
    x = (x - x.min()) / (x.max() - x.min())
    return x

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 15.0
    lrate = initial_lrate * math.pow(drop,  
                                     math.floor((1+epoch)/epochs_drop))
    return lrate

def addconvl(model, nf, ks, padding='valid', activation=True):	
	model.add(Conv2D(nf, (ks, ks), padding=padding, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
	model.add(BatchNormalization())
	if(activation):
	    model.add(LeakyReLU(alpha='0.1'))
	return model

def adddensel(model, size=1024, dropout=0.5):
	model.add(MaxoutDense(size))
	model.add(Dropout(dropout))
	model.add(LeakyReLU(alpha=0.1))
	return model

#from keras import backend as K
#def threshold_binary_accuracy(y_true, y_pred):
#    threshold = 0.80
#    if K.backend() == 'tensorflow':
#        return K.mean(K.equal(y_true, K.tf.cast(K.lesser(y_pred,threshold), y_true.dtype)))
#    else:
#        return K.mean(K.equal(y_true, K.lesser(y_pred,threshold)))

import math
def binaryaccuracy(y_true, y_pred, threshold=0.5): 
	threshold = math_ops.cast(threshold, y_pred.dtype)
	y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype) 
	return K.mean(math_ops.equal(y_true, y_pred), axis=-1)

def normalize_zscore(data):
    size = data.shape[0]

    for i in np.arange(0, size):
        for j in [0, 1, 2]:
            data[i, :, :, j] = zscore(data[i, :,:,j], axis=None)

    return data 
	
def histplot(history):
    hist = pd.DataFrame(history.history)
    hist.to_pickle('latest_hist.pk')

   # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
   # hist.plot(y=['loss', 'val_loss'], ax=ax1)
   # min_loss = hist['val_loss'].min()
   # ax1.hlines(min_loss, 0, len(hist), linestyle='dotted',
   #            label='min(val_loss) = {:.3f}'.format(min_loss))
   # ax1.legend(loc='upper right')
   # hist.plot(y=['acc', 'val_acc'], ax=ax2)
   # max_acc = hist['val_acc'].max()
   # ax2.hlines(max_acc, 0, len(hist), linestyle='dotted',
   #            label='max(val_acc) = {:.3f}'.format(max_acc))
   # ax2.legend(loc='lower right', fontsize='large')
   # fig.savefig('hist_{}.png'.format(time.time()))

def old_normalization_zscore(x_train):

        for idx, galaxy in enumerate(x_train):
                for jdx in [0,  1]:
                        mean = galaxy[...,jdx].mean()
                        std = galaxy[...,jdx].std()
                        x_train[idx, :, :, jdx] -= mean
                        x_train[idx, :, :, jdx] /= std

        return x_train

def normalization(x_train, x_test, x_val):

	means = np.zeros(x_train.shape[3])
	stds = np.zeros(x_train.shape[3])

	for idx, (mean, std) in enumerate(zip(means, stds)):

		means[idx] = x_train[..., idx].mean()
		stds[idx] = x_train[..., idx].std()

		x_train[..., idx] -= means[idx]
		x_train[..., idx] /= stds[idx]

		x_test[..., idx] -= means[idx]
		x_test[..., idx] /= stds[idx]

		#x_val[..., idx] -= means[idx]
		#x_val[..., idx] /= stds[idx]	

	return x_train, x_test, x_val, means, stds

def evaluate_candels(means, stds, X_train, cnnmodel):
    mergers =  np.load('../old_data/candels_cosmos_mergers.npy')[...,2:]
    wholecandels = np.load('../data/f_z05_nn_candels.npy')[...,2:]

    for idx, (mean, std) in enumerate(zip(means, stds)):
        X_train[..., idx] -= mean
        X_train[..., idx] /= std

        wholecandels[...,idx] -= mean
        wholecandels[...,idx] /= std

    wholecandelsnorm = cnnmodel.predict((wholecandels))
    training_pred = cnnmodel.predict(X_train)

    np.save('normed.npy', wholecandelsnorm)
    np.save('trainingpred.npy', training_pred)



