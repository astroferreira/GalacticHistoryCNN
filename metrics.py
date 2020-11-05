import numpy as np
import tensorflow as tf
import tensorflow.keras
import os
from itertools import *
import matplotlib
import io

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.models import Model

import tensorflow.python.keras.backend as K

class MergeNetLogger(Callback):

    def __init__(self,
                 dataset,
                 loss,
                 batch_size=32,
                 n_classes=2, 
                 writer=None,
                 log_images=False):
        
        super().__init__()
        
        self.dataset = dataset

        # SLOW but for when size are not known 
        self.batches = sum(1 for _ in self.dataset)
        print(self.batches)
        if log_images:
            self.batches = 227
        else:
            self.batches = 911

        self.loss = loss
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.writer = writer
        self.log_images = log_images
        self.upsilon = [0]
    
    def set_model(self, model):
        self.model = model         
       
    def on_epoch_end(self, epoch, logs={}):

        total = self.batches * self.batch_size
        
        val_pred = np.zeros((total, self.n_classes))
        val_true = np.zeros((total, self.n_classes))        

        for batch, (xVal, yVal) in enumerate(self.dataset):
            #xVal = xVal[:,:,:,0:3]
            if (batch == 0) & (epoch == 0):
                self.reference_X = xVal
                self.reference_y = yVal
        
            val_pred[batch * self.batch_size : (batch+1) * self.batch_size] = self.model.predict(xVal)
            val_true[batch * self.batch_size : (batch+1) * self.batch_size] = yVal.numpy()
        

        val_pred_binary = val_pred.round()

        pred = np.argmax(val_pred_binary, axis=1)
        true = np.argmax(val_true, axis=1)
        
        with self.writer.as_default():
            tf.summary.scalar(name='loss', data=self.loss(val_true, val_pred), step=epoch)
            tf.summary.scalar(name='lr', data=self.model.optimizer._decayed_lr(tf.float32), step=epoch)
            tf.summary.scalar(name='accuracy', data=accuracy_score(true, pred), step=epoch)

            # if epoch % 10 == 0:
            if self.log_images:
                self.log_confusion_matrix(true, pred, epoch)
                self.benchmark_reference(epoch)
            #         for i, layer in enumerate(self.model.layers):
            #             if 'dense' in layer.name:
            #                 tf.summary.histogram(f'{layer.name}_weights', layer.get_weights()[0], step=epoch)
            #                 tf.summary.histogram(f'{layer.name}_bias', layer.get_weights()[1], step=epoch)

            #     self.log_images_tb(epoch, val_pred_p, true)

        return

    def benchmark_reference(self, epoch):

        predictions = self.model.predict(self.reference_X)

        fig, axs = plt.subplots(4, 8, dpi=100)

        for entry, label, pred, ax in zip(self.reference_X.numpy(), self.reference_y.numpy(), predictions, axs.flat):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(10, 20, label[1])
            string_pred = str(np.round(pred[1], 2))
            ax.text(10, 110, string_pred)
            ax.imshow(np.sqrt(entry[:, :, 0]))

        plt.tight_layout()
        plt.subplots_adjust(hspace=-0.7, wspace=0)
        with self.writer.as_default():
            tf.summary.image('Reference Evolution', self.plot_to_image(fig), step=epoch)

    
    def log_confusion_matrix(self, true, pred, epoch):
        
        cm = confusion_matrix(true, pred)


        try:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]
        except:
            pass

        upsilon = np.round(np.min(np.diag(cm)), 3)
        if upsilon > np.max(self.upsilon):
            self.upsilon.append(upsilon)
            self.model.save(f'model_{upsilon}.h5')

        fig, ax = plt.subplots()
 
        
        n_classes = cm.shape[0]
        im_ = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        text_ = None
        
        cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

        text_ = np.empty_like(cm, dtype=object)

        thresh = (cm.max() + cm.min()) / 2.0

        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min

            text_cm = format(cm[i, j], '.2g')
            if cm.dtype.kind != 'f':
                text_d = format(cm[i, j], 'd')
                if len(text_d) < len(text_cm):
                    text_cm = text_d
        
            text_[i, j] = ax.text(
                j, i, text_cm,
                ha="center", va="center",
                color=color)

        
        display_labels = ['No MM', '1 or more MM']
        
        fig.colorbar(im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=display_labels,
               yticklabels=display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels())

        figure_ = fig
        ax_ = ax
        
        with self.writer.as_default():
            tf.summary.image('Confusion Matrix', self.plot_to_image(figure_), step=epoch)

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def gallery(self, array, ncols=3):
        nindex, height, width, intensity = array.shape

        nrows = nindex//ncols     
        assert nindex == nrows*ncols
        # want result.shape = (height*nrows, width*ncols, intensity)
        result = (array.reshape(nrows, ncols, height, width, intensity)
                .swapaxes(1,2)
                .reshape(height*nrows, width*ncols, intensity))
        return result
        
    def log_missclassifications(self, epoch, preds, trues, true_class_idx, threshold, summary_label):
        
        miss = np.where((preds > threshold) & (trues == true_class_idx))[0]
        if len(miss) > 64:
            batch_index = 0
            xVal, yVal, _ = self.generator[batch_index]
            n, h, w, c = xVal.shape
            miss_images = np.zeros((len(miss), h, w, c)) 
            for i, idx in enumerate(miss):
                """
                    when idx overflows current batch
                """
                if batch_index < idx // self.batch_size:
                    batch_index = idx // self.batch_size
                    xVal, yVal, _ = self.generator[batch_index]
                
                offset_index = idx - self.batch_size*batch_index
               
                miss_images[i] = xVal[offset_index]
                
            sort_order = np.argsort((-preds[miss]))
            miss_images = miss_images[sort_order,:,:,:]
            miss_images = miss_images[0:2*self.batch_size] 
            
            
            
            miss_grid = np.expand_dims(self.gallery(miss_images, ncols=8), axis=0)



            with self.writer.as_default():
                tf.summary.image(summary_label, miss_grid[:,:,:,:], max_outputs=1000, step=epoch)

    def log_images_tb(self, epoch, val_pred_p, true):

        self.log_missclassifications(epoch, val_pred_p.T[0], true, 1, 0.8, 'MM classified as NM')
        self.log_missclassifications(epoch, val_pred_p.T[1], true, 0, 0.8, 'NM classified as MM')

        X_first, yVal, _ = self.generator[0]
        augmentation_proxy = Model(inputs=self.model.inputs, outputs=self.model.layers[1].output)
        augmentation_output = augmentation_proxy(X_first, training=True)

        with self.writer.as_default():
            (nimgs, w, h, channels) = X_first.shape
            
            batch_proof_grid = np.expand_dims(self.gallery(X_first, ncols=8), axis=0)
            augmented_proof_grid = np.expand_dims(self.gallery(augmentation_output.numpy(), ncols=8), axis=0)

            tf.summary.image(f"Batch Reference Channel 1", batch_proof_grid[:,:,:,0:1], max_outputs=32, step=epoch)    
            tf.summary.image(f"Batch Reference Channel 2", batch_proof_grid[:,:,:,1:2], max_outputs=32, step=epoch)    
            tf.summary.image(f"Augmented Output Channel 1", augmented_proof_grid[:,:,:,0:1], max_outputs=32, step=epoch)    
            tf.summary.image(f"Augmented Output Channel 2", augmented_proof_grid[:,:,:,1:2], max_outputs=32, step=epoch)    
            

        for layer in self.model.layers:
            if 'cocozao' in layer.name:

                filters, biases = layer.get_weights()
                f_min, f_max = np.amin(filters), np.amax(filters)
                filters = (filters - f_min) / (f_max - f_min)
                f = filters.transpose(3,0,1,2)
        
                model_proxy = Model(inputs=self.model.inputs, outputs=layer.output)

                feature_maps = model_proxy.predict(X_first)

                feature_maps_grid = np.expand_dims(self.gallery(feature_maps, ncols=8), axis=0)
                #filters_grid = np.expand_dims(self.gallery(f, ncols=16), axis=0)


                with self.writer.as_default():
                    #tf.summary.image(f"Layer {layer.name}: Filters", filters_grid[:, :, :, 0:1], max_outputs=f.shape[0], step=epoch)
                    tf.summary.image(f"Layer {layer.name}: FeatureMaps", feature_maps_grid[:, :, :, 0:1], max_outputs=32, step=epoch)
                    #tf.summary.image(f"Layer {layer.name}: FeatureMaps Channel 1", feature_maps_grid[:, :, :, 0:1], max_outputs=32, step=epoch)
            

    
        
    def on_train_end(self, _):
        self.writer.close()
