import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image

import pandas as pd

from astropy.io import fits

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class MergerDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe, metadata, sample_weights=None, sample='train', batch_size=32, dim=(128, 128), n_channels=4, n_classes=2, shuffle=False, writer=None, path=None):
        'Initialization'
        self.dim = dim
        self.sample = sample

        if sample_weights is None:
            self.sample_weights = np.ones(dataframe.shape[0])
        else:
            self.sample_weights = sample_weights

        self.metadata = metadata
        self.batch_size = batch_size
        self.labels = dataframe['i_merger_count'].values
        self.dataframe = dataframe
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.skip_ids = np.array([])
        self.list_IDs = dataframe['IDs'].values
        self.writer = writer
        self.path = path
        self.on_epoch_end()
       

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = np.array([self.list_IDs[k] for k in indexes]).astype(int)

        # Generate data
        X, y, w = self.__data_generation(list_IDs_temp)

        return X, y, w

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes))
        w = np.empty((self.batch_size))
  
        #print(dim)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            rootname = self.dataframe.loc[self.dataframe['IDs'] == ID].rootname.values[0]
            #print(rootname)
            image = fits.getdata(f'{self.path}/FITS/{rootname}.fits')
            #image = fits.getdata(f'{self.path}/COMPOSITE/{rootname}.fits')
            
          
            X[i,] = image
                
            # Store class
            y[i] = self.labels[ID]# labels[ID] 
            w[i] = self.sample_weights[ID]
        
        return X, y, w