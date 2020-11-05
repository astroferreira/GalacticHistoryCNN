import os
import sys
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import time
import math
import h5py
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping, ModelCheckpoint


from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score as a
from sklearn import preprocessing
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta

from tensorflow.keras.applications import ResNet50, ResNet101

import logging
tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(4466)

import pandas as pd
import models as mdls

from metrics import MergeNetLogger
from generators import MergerDataGenerator
from datetime import datetime 



from sklearn.metrics import accuracy_score


class MergerNet(object):

    def __init__(self, data_path, callbacks=None):
        
        self.data_path = data_path
        self.callbacks = callbacks

        self.load_model()
        self.load_data()
        #self.load_generators()
        self.init_weights()

    def init_weights(self):

        weights, biases = self.model.layers[-1].get_weights()

        biases[0] = 0.5
        biases[1] = 0.5

        self.model.layers[-1].set_weights([weights, biases])

    def train(self, epochs):

        for epoch in range(epochs):         

            print("\nStart of epoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(self.training_dataset):
                
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = self.loss(y_batch_train, logits)
                                            
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                
                
                if step % 100 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                    print("Seen so far: %s samples" % ((step + 1) * 64))       
            
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.set_model(self.model)
                    callback.on_epoch_end(epoch)


            
    def load_model(self):
        self.model = mdls.base_model()
        initial_learning_rate = 0.1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                                    initial_learning_rate,
                                                                    decay_steps=1000,
                                                                    decay_rate=0.99,
                                                                    staircase=True)


        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr_schedule)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model.summary()
    
    #def load_generators(self):
        # self.train_generator = MergerDataGenerator(self.train_df, self.metadata,
        #                                      self.train_sample_weights, sample='train', 
        #                                      writer=file_writer_train, path=self.data_path)

        # self.test_generator = MergerDataGenerator(self.test_df, self.metadata,
        #                                      self.test_sample_weights, sample='test', 
        #                                      writer=file_writer_test, path=self.data_path)


    def load_data(self):

        def _parse_image_function(example_proto):
            
            image_feature_description = {
                'label': tf.io.FixedLenFeature([], tf.int64),
                'image_raw': tf.io.FixedLenFeature([], tf.string),
            }

            example = tf.io.parse_single_example(example_proto, image_feature_description)
            image = tf.io.decode_raw(example['image_raw'], out_type=np.float64)
            image = tf.reshape(image, (128, 128, 4))
            label = tf.one_hot(example['label'], 2, dtype=tf.int32)
            return image, label
        
        self.training_dataset = tf.data.TFRecordDataset(f'{self.data_path}/train.tfrecords')
        self.test_dataset = tf.data.TFRecordDataset(f'{self.data_path}/test.tfrecords')
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        self.training_dataset = self.training_dataset.with_options(ignore_order) 
        self.test_dataset = self.test_dataset.with_options(ignore_order)
        self.training_dataset = self.training_dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).shuffle(5000, reshuffle_each_iteration=True)
        self.test_dataset = self.test_dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32, drop_remainder=True)


    def set_callbacks(self, list):
        self.callbacks = list

if __name__ == "__main__":        
    
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer_train = tf.summary.create_file_writer(logdir + "/train")
    file_writer_test = tf.summary.create_file_writer(logdir + "/test")


    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    mergernet = MergerNet('data/BALANCED_ABOVE25')

    train_logger = MergeNetLogger(mergernet.training_dataset,
                                 mergernet.loss,
                                 writer=file_writer_train)

    test_logger = MergeNetLogger(mergernet.test_dataset,
                                 mergernet.loss,
                                 writer=file_writer_test,
                                 log_images=True)


    mergernet.set_callbacks([train_logger, test_logger])
    train_logger.set_model(mergernet.model)
    mergernet.train(10000)
