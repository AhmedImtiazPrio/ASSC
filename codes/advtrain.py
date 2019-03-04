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
from advutils import *
from AudioDataGenerator import BalancedAudioDataGenerator
from flipGradientTF import GradientReversal
from sklearn.preprocessing import StandardScaler
import math

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
    # parser.add_argument("--wakeReduction", type =bool,
    #                     help="Reduces wake class by a given percentage")
    # parser.add_argument("--s2Reduction", type = bool,
    #                     help="Reduces s1 class by a given percentage")
    parser.add_argument("--wakeRedSize", type=float,
                        help='the portion of wake data to be reduced')
    parser.add_argument("--s2RedSize", type= float,
                        help='the portion of s2 data to be reduced')
    parser.add_argument("--hp_lambda", type = float,
                        help = 'the update constant of the discriminator loss')
    parser.add_argument("--gamma", type = float,
                        help = 'the slope variable for hp_lambda sweep')


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
        batch_size = 64
        print("Training with %d minibatches" % (batch_size))

    if args.hp_lambda:
        print("harry potter lambda")
        hp_lambda = args.hp_lambda
    else:
        hp_lambda = 2.3

    if args.verbose:
        verbose = args.verbose
        print("Verbosity level %d" % (verbose))
    else:
        verbose = 2
    if args.comment:
        comment = args.comment
    else:
        comment = None

    # if args.wakeReduction:
    #     wakeReduction = args.wakeReduction
    # else:
    #     wakeReduction = False
    # if args.s2Reduction:
    #     s2Reduction = args.s2Reduction
    # else:
    #     s2Reduction = False
    if args.wakeRedSize:
        wakeRedSize = args.wakeRedSize
    else:
        wakeRedSize = 0.
    if args.s2RedSize:
        s2RedSize = args.s2RedSize
    else:
        s2RedSize = 0.
    if args.gamma:
        gamma = args.gamma
    else:
        gamma = 40.32
        print("The value of Gamma is 40.32")


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
    checkpoint_name = os.path.join(model_dir,log_name,'weights.{epoch:04d}-{val_clf_acc:.4f}.hdf5').replace('\\', '/')

#    results_file = os.path.join(os.getcwd(), '..', 'results.csv').replace('\\','/')

    results_file  = "E:/SleepWell/ASSC/results.csv"
    params = {

        'num_classes': 5,
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
        'dropout_rate': 0.45, #.5
        'dropout_rate_dense': 0.,
        'padding': 'valid',
        'activation_function': 'relu',
        'subsam': 2,
        'trainable': True,
        'lr': .0001, #.0001
        'lr_decay': 0.0, #1e-5, #1e-5
        'hp_lambda': 2.3, #hp_lambda
        'gamma': gamma
    }


    current_learning_rate= params['lr']


    df2 = pd.read_csv('E:/SleepWell/ASSC/data/lastpurifiedallDataChannel1.csv', header=None)
    df2.rename({3000: 'hyp', 3001: 'epoch', 3002: 'patID'}, axis="columns", inplace=True)

    #SC-task
    # trainX, valX, trainY, valY, pat_train, pat_val = patientSplitter('casetteID.csv', df2, 0.7, 39)

    #RS-task
    trainX, valX, trainY, valY, pat_train, pat_val = patientSplitter('randomizedIDsfinal.csv', df2, 0.7, 61)

    trainX -= np.mean(trainX,keepdims=True,axis=1)
    trainX /= (np.std(trainX,keepdims=True,axis=1) + K.epsilon())
    valX -= np.mean(valX,keepdims=True,axis=1)
    valX /= (np.std(valX,keepdims=True,axis=1) + K.epsilon())

    ############# Making 5 Class Data ############################


    for i in range(1, len(valY) + 1):
        if int(valY[i - 1]) == 4:
            valY[i - 1] = 3
    for j in range(1, len(valY) + 1):
        if int(valY[j - 1]) == 5:
            valY[j - 1] = 4
    for i in range(1, len(valY) + 1):
        if int(valY[i - 1]) == 6:
            valY[i - 1] = 5

    for i in range(1, len(trainY) + 1):
        if int(trainY[i - 1]) == 4:
            trainY[i - 1] = 3
    for j in range(1, len(trainY) + 1):
        if int(trainY[j - 1]) == 5:
            trainY[j - 1] = 4
    for i in range(1, len(trainY) + 1):
        if int(trainY[i - 1]) == 6:
            trainY[i - 1] = 5

    ###############################################################

    ## For softmax activation with categorical_crossentropy
    # mask = pat_train > 39
    # trainDom = to_categorical(mask.astype(int), 2)
    # mask = pat_val > 39
    # valDom = to_categorical(mask.astype(int), 2)
    print("Number of Patients in Train %d" % (len(np.unique(pat_train))))
    print("Number of Patients in Val %d" % (len(np.unique(pat_val))))
    # For sigmoid activation with binary_crossentropy
    mask = pat_train[:,0] >= 39
    trainDom = mask.astype(int)
    print(Counter(trainDom))
    mask = pat_val[:,0] >= 39
    valDom = mask.astype(int)
    print(Counter(valDom))

    # df2 = []
    del df2

    trainX, trainY = epoch_reduction(trainX, trainY, wakeRedSize, s2RedSize)


    # mean = np.mean(trainX)
    # std = np.std(trainX)
    #
    # valY = valY - mean
    # valY = valY / std


    # print("Dataframe has been loaded")
    dummytrainY = trainY-1
    dummytrainY = dummytrainY.astype(int)

    # print(Counter(trainY))
    trainY = to_categorical(trainY-1, params['num_classes'])
    valY = to_categorical(valY-1, params['num_classes'])
    trainX = np.expand_dims(trainX,axis=-1)
    valX = np.expand_dims(valX, axis=-1)

    eeg_length = 3000
    kernel_size= 16
    bias = False
    eps = 1.1e-5


    #### adverse-surreal

    K.clear_session()
    top_model = eegnet(**params)
    x = Flatten()(top_model.output)

    clf = Dense(params['num_classes'], activation='softmax',
                kernel_initializer=initializers.he_normal(seed=random_seed),
                name='clf',
                use_bias=True)(x)

    dann_in = GradientReversal(hp_lambda=params['hp_lambda'])(x) ## hp_lambda controls the effect of inverse gradient
    dsc = Dense(1, activation='sigmoid',
                kernel_initializer=initializers.he_normal(seed=random_seed),
                name='dsc',
                use_bias=True)(dann_in)

    model = Model(top_model.input, [clf,dsc])
    # model.summary()
    if load_path:
        model.load_weights(filepath=load_path, by_name=False)
    model_json = model.to_json()
    with open(os.path.join(model_dir, log_name, 'model.json').replace('\\','/'), "w") as json_file:
        json_file.write(model_json)


    ################### ADAM COMPILATION ##############
    model.compile(
                optimizer=opt(lr=params['lr'], epsilon=None, decay=params['lr_decay']),
                loss={'clf':'categorical_crossentropy','dsc':'binary_crossentropy'},
                metrics=['accuracy']
                 # loss_weights=[1,.5], ### Weighting the classifier loss by 1 and discriminator loss by .5
                  )  # মডেল কম্পাইলেশন। টেক্সটবুক আচরণ, অবশেষে
    ##################################################


    ################# SGD COMPILATION ################

    #sgd = optimizers.SGD(lr=params['lr'], decay=params['lr_decay'], momentum=0.9, nesterov=True)
    #model.compile(optimizer= sgd, loss= 'categorical_crossentropy', metrics=['accuracy'] )
    ##################################################


    print("model compilation: Done")
    modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                    monitor='val_clf_acc', save_best_only=True, mode='max')
    print("model Checkpoints: Loaded")

    tensdir = log_dir + "/" + log_name + "/"
    tensdir = tensdir.replace('/', "\\")

    tensbd = TensorBoard(log_dir=tensdir, batch_size=batch_size,write_grads=True,)

    # tensbd.set_model(model)

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
        lrate= params['lr']
        # if global_epoch_counter>10:
        #     lrate=params['lr']/10
        #     if global_epoch_counter>20:
        #         lrate=params['lr']/100
        #         # if global_epoch_counter>30:
        #         #     lrate=params['lr']/1000
        return lrate
    lrate = LearningRateScheduler(step_decay)


    def f_hp_decay(global_epoch_counter=global_epoch_counter, params=params):

        print("global_epoch_counter")
        print(global_epoch_counter)

        gamma =  params['gamma']
        p = (global_epoch_counter) / params['epochs']
        hp_lambda =  (4 / (1 + 3*(math.e ** (- gamma * p)))) - 1  # 3 porjonto jaabe
        # hp_lambda = hp_lambda * (params['hp_decay_const'] ** global_epoch_counter)
        params['hp_lambda'] = hp_lambda
        print(hp_lambda)
        return hp_lambda


    class hpRateScheduler(Callback):
        """Learning rate scheduler.
        # Arguments
            schedule: a function that takes an epoch index as input
                (integer, indexed from 0) and current learning rate
                and returns a new learning rate as output (float).
            verbose: int. 0: quiet, 1: update messages.
        """

        def __init__(self, schedule, params, verbose=0):
            super(hpRateScheduler, self).__init__()
            self.schedule = schedule
            self.verbose = verbose
            self.params = params

        def on_epoch_begin(self, epoch, params=params, logs=None):
            # if not hasattr(self.model.optimizer, 'hp_lambda'):
            #     raise ValueError('Optimizer must have a "lr" attribute.')
            #a = GradientReversal( hp_lambda = params['hp_lambda'])
            hp_lambda = self.model.layers[-3].hp_lambda  # float(K.get_value(self.model.optimizer.lr))
            try:  # new API
                hp_lambda = self.schedule(epoch, hp_lambda)
            except TypeError:  # old API for backward compatibility
                hp_lambda = self.schedule(epoch)
            if not isinstance(hp_lambda, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function '
                                 'should be float.')
            # K.set_value(self.model.layers[-3].hp_lambda, hp_lambda)
            self.model.layers[-3].hp_lambda = hp_lambda
            if self.verbose > 0:
                print('\nEpoch %05d: HP setting hp_lambda '
                      'rate to %s.' % (epoch + 1, hp_lambda))

        def on_epoch_end(self, epoch, params=params, logs=None):
            logs = logs or {}
            logs['hp_lambda'] = hp_lambda


    hprate = hpRateScheduler(f_hp_decay, params)

    try:

        datagen = BalancedAudioDataGenerator(
                                      # shift=.1,
                                      #roll_range=.15,
                                     # fill_mode='reflect',
                                     # featurewise_center=True,
                                     # zoom_range=.1,
                                     # zca_whitening=True,
                                     #  samplewise_center=True,
                                     #  samplewise_std_normalization=True,
                                             )

        # valgen = AudioDataGenerator(
        #      # fill_mode='reflect',
        #      # featurewise_center=True,
        #      # zoom_range=.2,
        #      # zca_whitening=True,
        #       #roll_range=.1, রোল বন্ধ, যাতে সব সময় একই ডাটার উপরে ভ্যালিডেশন হয়।
        #       samplewise_center=True,
        #       samplewise_std_normalization=True,
        # )

        meta_labels = dummytrainY
        print("meta_labels")
        print(np.unique(meta_labels))
        print(pat_train.shape)
        for idx, each in enumerate(np.unique(dummytrainY)):
           meta_labels[np.where(np.logical_and(np.asarray(pat_train[:,0]) >= 39, np.asarray(dummytrainY) == each))] = 5 + idx

        flow = datagen.flow(trainX, [trainY, trainDom],
                            meta_label=meta_labels,
                            batch_size=params['batch_size'], shuffle=True, seed=params['random_seed'])
        model.fit_generator(flow,
                            #steps_per_epoch= sum(np.asarray(meta_labels==9)) // flow.chunk_size,
                             steps_per_epoch=1000,
                            epochs=params['epochs'],
                            validation_data=(valX, [valY,valDom]),
                            #validation_data=valgen.flow(valX, valY, batch_size=params['batch_size'],
                            #                            seed=params['random_seed']),
                            callbacks=[modelcheckpnt, log_metrics(valX, [valY, valDom], pat_val, params, patlogDirectory, global_epoch_counter),
                                       csv_logger, tensbd, lrate,
                                       #hprate
                                        ],
                            #class_weight=params['class_weight']
                            )


    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        results_log(results_file=results_file, log_dir=log_dir, log_name= log_name, params=params)
