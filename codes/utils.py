import numpy as np
from keras.callbacks import Callback
from keras import backend as K
from sklearn.metrics import confusion_matrix

def compute_weight(Y, classes):
    num_samples = len(Y)
    n_classes = len(classes)
    num_bin = np.bincount(Y[:, 0])
    class_weights = {i: (num_samples / (n_classes * num_bin[i])) for i in range(6)}
    return class_weights

class log_metrics(Callback):

    def __init__(self, valX, valY, patID, **kwargs):
        self.valY = valY
        self.valX = valX
        self.patID = patID

    def on_epoch_end(self, epoch, logs):

        if logs is not None:
            predY = self.model.predict(self.valX, verbose=0)
            predY = np.argmax(predY, axis=-1)

            ## Enter metric calculations per patient here

            #### Learning Rate logging for Adam ###

            lr = self.model.optimizer.lr
            if self.model.optimizer.initial_decay > 0:
                lr *= (1. / (1. + self.model.optimizer.decay * K.cast(self.model.optimizer.iterations,
                                                                      K.dtype(self.model.optimizer.decay))))
            t = K.cast(self.model.optimizer.iterations, K.floatx()) + 1
            lr_t = lr * (
                    K.sqrt(1. - K.pow(self.model.optimizer.beta_2, t)) / (1. - K.pow(self.model.optimizer.beta_1, t)))
            logs['lr'] = np.array(float(K.get_value(lr_t)))
