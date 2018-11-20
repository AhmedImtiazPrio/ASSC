import numpy as np
from keras.callbacks import Callback
from keras import backend as K
import pandas as pd

# class_weight={0:3.3359,1:0.3368,2:3.0813,3:2.7868,4:0.7300,5:1.4757}
def compute_weight(Y, classes):
    num_samples = len(Y)
    n_classes = len(classes)
    num_bin = np.bincount(Y[:, 0])
    class_weights = {i: (num_samples / (n_classes * num_bin[i])) for i in range(6)}
    return class_weights

def results_log(filepath, params):

    df = pd.read_csv(filepath)
    df1 = pd.read_csv()

    ## logging pipeline here ##

    print()

class log_metrics(Callback):
    '''Custom callback to add new elements to epoch end logs and tensorflow graphs

        Callback called on_epoch_end to calcualte validations set metrics using meta
        information (Patient ID, channel ID, etc). End of epoch metrics are added to
        `logs` which are later used by the tensorboard callback.
        log_metrics should be specified in the callback list before tensorboard callback
        '''

    def __init__(self, valX, valY, patID, **kwargs):
        self.valY = valY
        self.valX = valX
        self.patID = patID
        super(log_metrics,self).__init__(**kwargs)

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


