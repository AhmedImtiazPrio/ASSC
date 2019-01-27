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
from imblearn.datasets import make_imbalance
from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss
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
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import pandas as pd

from modules import *
from utils import *
from AudioDataGenerator import *

from sklearn.preprocessing import StandardScaler


import xgboost as xgb

global_epoch_counter = 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("fold",
                        help="csvfile to use")
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
    parser.add_argument("--s1Reduction", type = bool,
                        help="Reduces s1 class by a given percentage")
    parser.add_argument("--wakeRedSize", type=float,
                        help='the portion of wake data to be reduced')
    parser.add_argument("--s1RedSize", type= float,
                        help='the portion of s1 data to be reduced')


    args = parser.parse_args()
    print("%s selected" % (args.fold))
    foldname = args.fold

    if args.seed:  # if random seed is specified
        print("Random seed specified as %d" % (args.seed))
        random_seed = args.seed
    else:
        random_seed = 1

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

    if args.wakeReduction:
        wakeReduction = args.wakeReduction
    if args.s1Reduction:
        s1Reduction = args.s1Reduction
    if args.wakeRedSize:
        wakeRedSize = args.wakeRedSize
    if args.s1RedSize:
        s1RedSize = args.s1RedSize


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

        'num_classes': 6,
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
        'maxnorm': 400000000000.,
        'dropout_rate': 0.6, #.5
        'dropout_rate_dense': 0.,
        'padding': 'valid',
        'activation_function': 'relu',
        'subsam': 2,
        'trainable': True,
        'lr': .0007, #.0001
        'lr_decay': 0 #1e-5, #1e-5
    }






    df2 = pd.read_csv('E:/SleepWell/ASSC/data/lastpurifiedallDataChannel1.csv', header=None)
    df2.rename({3000: 'hyp', 3001: 'epoch', 3002: 'patID'}, axis="columns", inplace=True)

    trainX, valX, trainY, valY, pat_train, pat_val = patientSplitter('casetteID.csv', df2, 0.7, 39)
    # trainX = standardnormalization(trainX)
    # valX = standardnormalization(valX)
    df2 = []

    trainX, trainY = epoch_reduction(trainX, trainY, wakeReduction, wakeRedSize, s1Reduction, s1RedSize)


    # mean = np.mean(trainX)
    # std = np.std(trainX)
    #
    # valY = valY - mean
    # valY = valY / std


    print("Dataframe has been loaded")
    dummytrainY = trainY-1
    dummytrainY = dummytrainY.astype(int)

    print(Counter(trainY))
    trainY = to_categorical(trainY-1, params['num_classes'])
    valY = to_categorical(valY-1, params['num_classes'])
    trainX = np.expand_dims(trainX,axis=-1)
    valX = np.expand_dims(valX, axis=-1)

    eeg_length = 3000
    kernel_size= 16
    bias = False
    maxnorm=4
    eps = 1.1e-5

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
    model.compile(optimizer=opt(lr=params['lr'], epsilon=None, decay=params['lr_decay']), loss='categorical_crossentropy', metrics=['accuracy'])  # মডেল কম্পাইলেশন। টেক্সটবুক আচরণ, অবশেষে
    print("model compilation: Done")
    modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                    monitor='val_acc', save_best_only=False, mode='max')
    print("model Checkpoints: Loaded")

    tensdir = log_dir + "/" + log_name + "/"
    tensdir = tensdir.replace('/', "\\")

    tensbd = TensorBoard(log_dir=tensdir, batch_size=batch_size, write_grads=True,)

    print("Tensorboard initialization: Done")

    patlogDirectory = log_dir+'/' + log_name + '/'
    trainingCSVdirectory = log_dir+'/'+log_name+'/'+'training.csv'
    csv_logger = CSVLogger(trainingCSVdirectory)
    print("csv logger: Activated")
    if args.classweights:
        params['class_weight'] = compute_weight(dummytrainY, np.unique(dummytrainY))
    else:
        params['class_weight'] = dict(zip(np.r_[0:params['num_classes']], np.ones(params['num_classes'])))

    print("model dot fit: Started")

    def step_decay(global_epoch_counter):
        # initial_lrate = params['lr']
        # drop = params['lr']*9/10
        # epochs_drop = 10.0
        # lrate = initial_lrate * math.pow(drop, math.floor((1 + global_epoch_counter) / epochs_drop))


        if global_epoch_counter<11:
            lrate = params['lr']
        if global_epoch_counter>10:
            lrate = params['lr']/10

        return lrate


    lrate = LearningRateScheduler(step_decay)

    try:

        datagen = AudioDataGenerator(
                                      shift=.1,
                                      roll_range=.15,
                                     # fill_mode='reflect',
                                     # featurewise_center=True,
                                     # zoom_range=.1,
                                     # zca_whitening=True,
                                      samplewise_center=True,
                                      samplewise_std_normalization=True,
                                             )

        valgen = AudioDataGenerator(
             # fill_mode='reflect',
             # featurewise_center=True,
             # zoom_range=.2,
             # zca_whitening=True,
              #roll_range=.1, রোল বন্ধ, যাতে সব সময় একই ডাটার উপরে ভ্যালিডেশন হয়।
              samplewise_center=True,
              samplewise_std_normalization=True,
        )
        print("printing the weights")
        print(compute_weight(dummytrainY, np.unique(dummytrainY)))


        #training_generator, steps_per_epoch = BalancedBatchGenerator(trainX, trainY, batch_size=params['batch_size'], random_state = 42)
        #
        # model.fit_generator(generator=training_generator, steps_per_epoch = steps_per_epoch, epochs = params['epochs'], verbose = 0,
        #                     callbacks=[modelcheckpnt, log_metrics(valX, valY, pat_val, patlogDirectory, global_epoch_counter),
        #                                 csv_logger, tensbd], validation_data=(valX, valY))



        model.fit_generator(datagen.flow(trainX, trainY, batch_size=params['batch_size'], shuffle=True, seed=params['random_seed']),
                            steps_per_epoch=len(trainX) // params['batch_size'],
                            epochs=params['epochs'],
                            callbacks=[modelcheckpnt, log_metrics(valX, valY, pat_val, patlogDirectory, global_epoch_counter),
                                       csv_logger, tensbd, lrate],
                            validation_data=valgen.flow(valX, valY, batch_size= params['batch_size'], seed=params['random_seed']),
                            class_weight=params['class_weight']
                            )

        #
        # model.fit(trainX, trainY, validation_data=(valX, valY),
        #        callbacks=[modelcheckpnt, log_metrics(valX, valY, pat_val, patlogDirectory, global_epoch_counter), csv_logger, tensbd],
        #       batch_size=64, epochs=params['epochs'])
        results_log(results_file=results_file, log_dir=log_dir, log_name= log_name, params=params)

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        results_log(results_file=results_file, log_dir=log_dir, log_name= log_name, params=params)
