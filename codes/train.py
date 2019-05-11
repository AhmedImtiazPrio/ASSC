from __future__ import print_function, absolute_import, division
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))
import numpy as np
from collections import Counter
np.random.seed(1)
from tensorflow import set_random_seed
# from imblearn.datasets import make_imbalance
# from imblearn.keras import BalancedBatchGenerator
# from imblearn.under_sampling import NearMiss
from keras.callbacks import LearningRateScheduler

set_random_seed(1)
from datetime import datetime
import argparse
import os
import tables
from keras.utils import to_categorical, plot_model
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adamax as opt
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import pandas as pd

from modules import *
from utils import *
from AudioDataGenerator import *

from sklearn.preprocessing import StandardScaler

global_epoch_counter = 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("fold",
                        help="data csvfile to use")
    parser.add_argument("--num_class", type=int,
                        help="One of {5,6}- Classification method")
    parser.add_argument("--seed", type=int,
                        help="Random seed")
    parser.add_argument("--loadmodel",
                        help="load previous model checkpoint for retraining (Enter absolute path)")
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int,
                        help="number of minibatches to take during each backwardpass preferably multiple of 2")
    parser.add_argument("--verbose", type=int, choices=[1, 2],
                        help="Verbosity mode. 1 = progress bar, 2 = one line per epoch (default 2)")
    parser.add_argument("--classweights", type=bool,
                        help="if True, class weights are added")
    parser.add_argument("--comment",
                        help="Add comments to the log files")
    parser.add_argument("--wakeReduction", type =bool,
                        help="Reduces wake class by a given percentage")
    parser.add_argument("--s2Reduction", type = bool,
                        help="Reduces s1 class by a given percentage")
    parser.add_argument("--wakeRedSize", type=float,
                        help='the portion of wake data to be reduced')
    parser.add_argument("--s2RedSize", type= float,
                        help='the portion of s2 data to be reduced')


    args = parser.parse_args()
    print("%s selected" % (args.fold))
    foldname = args.fold

    if args.seed:  # if random seed is specified
        print("Random seed specified as %d" % (args.seed))
        random_seed = args.seed
    else:
        random_seed = 1
    if args.num_class:
        num_class = args.num_class
    else:
        num_class =5

    if args.loadmodel:  # If a previously trained model is loaded for retraining
        load_path = args.loadmodel  #### path to model to be loaded

        idx = load_path.find("weights")
        initial_epoch = int(load_path[idx + 8:idx + 8 + 4])

        print("%s model loaded\nInitial epoch is %d" % (args.loadmodel, initial_epoch))
    else:
        print("no model specified, using initializer to initialize weights")
        initial_epoch = 0
        load_path = False

    if args.epochs:  # if number of training epochs is specified
        print("Training for %d epochs" % (args.epochs))
        epochs = args.epochs
    else:
        epochs = 200
        print("Training for %d epochs" % (epochs))

    if args.batch_size:  # if batch_size is specified
        print("Training with %d samples per minibatch" % (args.batch_size))
        batch_size = args.batch_size
    else:
        batch_size = 1024
        print("Training with %d minibatches" % (batch_size))

    if args.verbose:
        verbose = args.verbose
        print("Verbosity level %d" % (verbose))
    else:
        verbose = 2
    if args.comment:
        comment = args.comment
    else:
        comment = None


    model_dir = os.path.join(os.getcwd(),'..','models').replace('\\', '/')
    fold_dir = os.path.join(os.getcwd(),'..','data').replace('\\', '/')
    log_dir = os.path.join(os.getcwd(),'..','logs').replace('\\', '/')
    log_name = foldname + '_' + str(datetime.now()).replace(':','-')

    if not os.path.exists(os.path.join(model_dir, log_name).replace('\\', '/')):
        new_dir = (os.path.join(model_dir, log_name).replace('\\', '/'))
        print(new_dir)
        os.makedirs(new_dir)

    if not os.path.exists(os.path.join(log_dir, log_name).replace('\\', '/')):
        new_dir = os.path.join(log_dir, log_name).replace('\\', '/')
        print(new_dir)
        os.makedirs(new_dir)

    checkpoint_name = os.path.join(model_dir,log_name,'weights.{epoch:04d}-{val_acc:.4f}.hdf5').replace('\\', '/')
    results_file = os.path.join(os.getcwd(), '..', 'results.csv').replace('\\','/')

    params = {

        'num_classes': num_class,
        'batch_size': batch_size,
        'epochs': epochs,
        'aafoldname': foldname,
        'random_seed': random_seed,
        'load_path': load_path,
        'shuffle': True,
        'initial_epoch': initial_epoch,
        'eeg_length': 3000,
        'kernel_size': 16,
        'bias': True,
        'maxnorm': 400000000000., ## No maxnorm constraint
        'dropout_rate': 0.45, #.5
        'dropout_rate_dense': 0.,
        'padding': 'valid',
        'activation_function': 'relu',
        'subsam': 2,
        'trainable': True,
        'lr': .001, #.0001
        'lr_decay': 0.0 #1e-5, #1e-5
    }


    current_learning_rate= params['lr']

    df = pd.read_csv(os.path.join(fold_dir,foldname).replace('\\', '/'), header=None)
    trainX, valX, trainY, valY, pat_train, pat_val = patientSplitter(df,task='RS',stages=params['num_classes'])
    del df

    print("Data loaded")
    
    if args.classweights:
        params['class_weight'] = compute_weight(trainY.astype(int), np.unique(trainY.astype(int)))
    else:
        params['class_weight'] = dict(zip(np.r_[0:params['num_classes']], np.ones(params['num_classes'])))

    print('Classwise data in train',Counter(trainY))
	
    trainY = to_categorical(trainY)
    valY = to_categorical(valY)
    trainX = np.expand_dims(trainX,axis=-1)
    valX = np.expand_dims(valX, axis=-1)

    K.clear_session()
    top_model = eegnet(**params)
    x = Flatten()(top_model.output)
    x = Dense(params['num_classes'], activation='softmax', kernel_initializer=initializers.he_normal(seed=random_seed),
              kernel_constraint=max_norm(params['maxnorm']), use_bias=True)(x)

    model = Model(top_model.input, x)
    model.summary()
    if load_path:
        model.load_weights(filepath=load_path, by_name=False)
    model_json = model.to_json()
    with open(os.path.join(model_dir, log_name, 'model.json').replace('\\','/'), "w") as json_file:
        json_file.write(model_json)

    model.compile(optimizer=opt(lr=params['lr'], epsilon=None, decay=params['lr_decay']),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    ####### Callbacks #######

    modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                    monitor='val_acc', save_best_only=True, mode='max')

    tensdir = log_dir + "/" + log_name + "/"
    tensdir = tensdir.replace('/', "\\")
    tensbd = TensorBoard(log_dir=tensdir, batch_size=batch_size, write_grads=True,)
    patlogDirectory = log_dir +'/'+ log_name +'/'
    trainingCSVdirectory = log_dir +'/'+ log_name +'/'+ 'training.csv'
    csv_logger = CSVLogger(trainingCSVdirectory)


    ####### Training #########

    try:

        datagen = AudioDataGenerator(
                                      # shift=.1,
                                      roll_range=.15,
                                      samplewise_center=True,
                                      samplewise_std_normalization=True,
                                             )

        valgen = AudioDataGenerator(

              samplewise_center=True,
              samplewise_std_normalization=True,
        )

        model.fit_generator(datagen.flow(trainX, trainY, batch_size=params['batch_size'], shuffle=True, seed=params['random_seed']),
                            steps_per_epoch=len(trainX) // params['batch_size'],
                            epochs=params['epochs'],
                            validation_data=valgen.flow(valX, valY, batch_size=params['batch_size'],
                                                        seed=params['random_seed']),
                            callbacks=[modelcheckpnt, log_metrics(valX, valY, pat_val, patlogDirectory, global_epoch_counter),
                                       csv_logger, tensbd, lrate],
                            class_weight=params['class_weight']
                            )

        results_log(results_file=results_file, log_dir=log_dir, log_name= log_name, params=params)

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        results_log(results_file=results_file, log_dir=log_dir, log_name= log_name, params=params)
