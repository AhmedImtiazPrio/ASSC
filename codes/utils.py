from __future__ import print_function, absolute_import, division
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from datetime import datetime
import argparse
import os
import tables
from keras.utils import to_categorical, plot_model
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adamax as opt
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, Callback
import pandas as pd

from modules import *
#from utils import *





##############remove these imports later.




# import numpy as np
# from keras.callbacks import Callback
# from keras import backend as K
# import pandas as pd

#class_weight={0:3.3359,1:0.3368,2:3.0813,3:2.7868,4:0.7300,5:1.4757}

########### এই টা জিরো থেকে শুরু করলে কাজ করে।
def compute_weight(Y, classes):
    num_samples = len(Y)
    n_classes = len(classes)
    num_bin = np.bincount(Y[:, 0])
    class_weights = {i: (num_samples / (n_classes * num_bin[i])) for i in range(6)}
    return class_weights

def patientSplitter(randomIDfile,df2,split_portion):
    import pandas as pd
    
    df1 = pd.read_csv(randomIDfile,header=None)
    split_portion_numer=int(split_portion*61)
    
    train_pat_list = [int(each) for each in df1.iloc[:split_portion_numer].values]
    test_pat_list = [int(each) for each in df1.iloc[split_portion_numer:].values]
    print(test_pat_list)
    df3 = []
    df4 = []
    for pat_ID in train_pat_list:
        df3.append(df2[df2.patID == pat_ID].values)
        print(pat_ID)
    for pat_ID in test_pat_list:
        df4.append(df2[df2.patID == pat_ID].values) 
        print(pat_ID)
    del df2
    df3 = pd.np.vstack(df3)
    df4 = pd.np.vstack(df4)
    X_train = df3[:, :3000]
    X_test= df4[:, :3000]
    Y_train= df3[:,3000]
    Y_test= df4[:,3000]
    pat_train=df3[:, 3002:3003]
    pat_test= df4[:, 3002:3003]
    
    del pd
    return X_train,X_test,Y_train,Y_test,pat_train,pat_test	


    #### রেজাল্ট এর লগিং এখানে। একটা কল ব্যাক দিয়ে সিএসভি ফাইলে গিয়ে ডাম্প করে আসতে হবে।

# def results_log(results_file, log_dir, log_name, params):
#     print('baal')
#
#


def results_log(results_file, log_dir, log_name, params):
    df = pd.read_csv(results_file)
    df1 = pd.read_csv(os.path.join(log_dir, log_name, 'training.csv').replace('\\', '/'))
    terminalrow = df1['']

    # new_entry = {'Filename': '*'+params['log_name'],
    #              'num_classes': params['num_classes'],  ### automate; number of classes depends on data fold
    #              'batch_size': params['batch_size'],
    #              'epochs': params['epochs'],
    #              'foldname': params['foldname'],
    #              'random_seed': params['random_seed'],
    #              'load_path': params['load_path'],
    #              #'shuffle' : params['shuffle'],
    #              'initial_epoch': params['initial_epoch'],
    #              'eeg_length' : params['eeg_length'],
    #              'kernel_size': 16,
    #              #'bias': params['bias'],
    #              'maxnorm': params['maxnorm'],
    #              'dropout_rate': params['dropout_rate'],
    #              'dropout_rate_dense': params['dropout_rate_dense'],
    #              'padding': params['padding'],
    #              'activation_function': params['activation_function'],
    #              'subsam': params['subsam'],
    #              #'trainable' : params['trainable'],
    #              'learningRate': params['lr'],
    #              'learningRateDecay': params['lr_decay'],
    #              'classWeights' : params['class_weight'],
    #              'numberofClasses' : params['num_classes'],
    #
    #             }

    new_entry - params
    index, _ = df.shape
    new_entry = pd.DataFrame(new_entry, index = [index])
    print(new_entry)
    #df2 = pd.concat([df, new_entry], axis= 0)
    df2.to_csv(results_file, index=False)
    #df2.tail()
    print("Saving to results.csv")
    # df = pd.read_csv(filepath)
    # df1 = pd.read_csv('trin.csv') #training.csv

    #logging pipeline here ##


    #print()


    ########## লগ ম্যাট্রিক্স বানানো।  -_-  #################

    '''Custom callback to add new elements to epoch end logs and tensorflow graphs

            # Callback called on_epoch_end to calcualte validations set metrics using meta
            # information (Patient ID, channel ID, etc). End of epoch metrics are added to
            # `logs` which are later used by the tensorboard callback.
            # log_metrics should be specified in the callback list before tensorboard callback
            # '''


class log_metrics( Callback):


    ###### লগ ম্যাট্রিক্স ক্লাসের ইনিশিয়ালাইজার ##################
    #### চার টা জিনিস নিচ্ছে সে। এক্স এর মান, ওয়াই এর মান, পেশেন্ট আইডি। আর কাজ করানোর জন্য খরগোশ।
    def __init__(self, valX, valY, patID, **kwargs):
        self.valY = np.argmax(valY, axis=-1)
        self.valX = valX
        self.patID = patID
        super(log_metrics,self).__init__(**kwargs) #### এই সুপার টা এই ফাংশনের পার্ট না হয়ে মেইন ক্লাসের পার্ট হয়ে গেল। 



    def on_epoch_end(self, epoch, logs):

        if logs is not None:
            # predY = self.model.predict(self.valX, verbose=0)
            # predY = np.argmax(predY, axis=-1)

            # Enter metric calculations per patient here
            #
            # acc = .9
            # logs['acc']  = acc

            ### Learning Rate logging for Adam ###

            lr = self.model.optimizer.lr
            if self.model.optimizer.initial_decay > 0:
                lr *= (1. / (1. + self.model.optimizer.decay * K.cast(self.model.optimizer.iterations, K.dtype(self.model.optimizer.decay))))
            t = K.cast(self.model.optimizer.iterations, K.floatx()) + 1
            lr_t = lr * (K.sqrt(1. - K.pow(self.model.optimizer.beta_2, t)) / (1. - K.pow(self.model.optimizer.beta_1, t)))
            logs['lr'] = np.array(float(K.get_value(lr_t)))


