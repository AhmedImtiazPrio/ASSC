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
from keras.losses import categorical_crossentropy
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
import keras
from hyperopt import hp, Trials, fmin, tpe
from modules import *
from advutils import *
from AudioDataGenerator import BalancedAudioDataGenerator
from flipGradientTF import GradientReversal
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
        hp_lambda = 1

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
            'lr': .001, #.0001
            'lr_decay': 0.0, #1e-5, #1e-5
            'hp_lambda':  hp_lambda
            }




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
    # log_name = foldname + '_' + str(datetime.now()).replace(':', '-')
    # # model_dir = os.path.join(os.getcwd(), '..', 'models').replace('\\', '/')
    # # fold_dir = os.path.join(os.getcwd(), '..', 'data').replace('\\', '/')
    # # log_dir = os.path.join(os.getcwd(), '..', 'logs').replace('\\', '/')
    # #
    # model_dir = os.path.join(os.getcwd(), '..', 'models').replace('\\', '/')
    # fold_dir = os.path.join(os.getcwd(), '..', 'data').replace('\\', '/')
    # # log_dir = os.path.join(os.getcwd(), '..', 'logs').replace('\\', '/')

    K.clear_session()
    top_model = eegnet(**params)
    x = Flatten()(top_model.output)

    clf = Dense(params['num_classes'], activation='softmax',
                kernel_initializer=initializers.he_normal(seed=random_seed),
                name='clf',
                use_bias=True)(x)

    dann_in = GradientReversal(hp_lambda=params['hp_lambda'])(x)
    dsc = Dense(1, activation='sigmoid',
                kernel_initializer=initializers.he_normal(seed=random_seed),
                name='dsc',
                use_bias=True)(dann_in)

    model = Model(top_model.input, [clf,dsc])
    # model.summary()
    # if load_path:
    #     model.load_weights(filepath=load_path, by_name=False)
    # model_json = model.to_json()
    # with open(os.path.join(model_dir, log_name, 'model.json').replace('\\','/'), "w") as json_file:
    #     json_file.write(model_json)



    print("model compilation: Done")


    def objective(args, params=params):

        from keras.losses import categorical_crossentropy
        # for i in range(len(args)):
        #     print(i)
        for each in args:
            print(each)
        log_name = "hyperopt"+ '_' + str(datetime.now()).replace(':', '-')
        model_dir = os.path.join(os.getcwd(), '..', 'models').replace('\\', '/')
        fold_dir = os.path.join(os.getcwd(), '..', 'data').replace('\\', '/')
        log_dir = os.path.join(os.getcwd(), '..', 'logs').replace('\\', '/')

        model_dir = os.path.join(os.getcwd(), '..', 'models').replace('\\', '/')
        fold_dir = os.path.join(os.getcwd(), '..', 'data').replace('\\', '/')
        log_dir = os.path.join(os.getcwd(), '..', 'logs').replace('\\', '/')


        if not os.path.exists(os.path.join(model_dir, log_name).replace('\\', '/')):
            new_dir = (os.path.join(model_dir, log_name).replace('\\', '/'))
            print(new_dir)
            os.makedirs(new_dir)
        if not os.path.exists(os.path.join(log_dir, log_name).replace('\\', '/')):
            new_dir = os.path.join(log_dir, log_name).replace('\\', '/')
            print(new_dir)
            os.makedirs(new_dir)

        checkpoint_name = os.path.join(model_dir, log_name, 'weights.{epoch:04d}-{val_clf_acc:.4f}.hdf5').replace('\\',
                                                                                                                  '/')

        results_file = os.path.join(os.getcwd(), '..', 'results.csv').replace('\\', '/')

        params['dropout_rate'] = args[1]
        params['lr'] = args[2]
        params['hp_lambda'] = args[0]

        # params = {
        #     'dropout_rate': args[1],  # 0.45, #.5
        #     'lr': args[2],  # .0001
        #     'hp_lambda': args[0],
        # }
        current_learning_rate = params['lr']

        model.compile(
            optimizer=opt(lr=params['lr'], epsilon=None, decay=params['lr_decay']),
            loss={'clf': 'categorical_crossentropy', 'dsc': 'binary_crossentropy'},
            metrics=['accuracy']
        )





        modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                        monitor='val_clf_acc', save_best_only=True, mode='max')
        print("model Checkpoints: Loaded")
        tensdir = log_dir + "/" + "hyperopt-{}".format(args) + "/"
        tensdir = tensdir.replace('/', "\\")
        tensbd = TensorBoard(log_dir=tensdir, batch_size=batch_size, write_grads=True, )
        print("Tensorboard initialization: Done")
        patlogDirectory = log_dir + '/' + log_name + '/'
        trainingCSVdirectory = log_dir + '/' + log_name + '/' + 'training.csv'
        csv_logger = CSVLogger(trainingCSVdirectory)
        print("csv logger: Activated")
        # if args.classweights:
        #     params['class_weight'] = compute_weight(dummytrainY, np.unique(dummytrainY))
        # else:
        #     params['class_weight'] = dict(zip(np.r_[0:params['num_classes']], np.ones(params['num_classes'])))

        print("model dot fit: Started")

        def step_decay(global_epoch_counter):
            lrate = params['lr']
            # if global_epoch_counter>10:
            #     lrate=params['lr']/10
            #     if global_epoch_counter>20:
            #         lrate=params['lr']/100
            #         # if global_epoch_counter>30:
            #         #     lrate=params['lr']/1000
            return lrate

        lrate = LearningRateScheduler(step_decay)

        try:

            datagen = BalancedAudioDataGenerator()

            flow = datagen.flow(trainX, [trainY, trainDom], target_label=1, batch_size=params['batch_size'],
                                shuffle=True, seed=params['random_seed'])
            model.fit_generator(flow,
                                steps_per_epoch=len(trainDom[trainDom == 0]) // flow.chunk_size,
                                # steps_per_epoch=4,
                                epochs=params['epochs'],
                                validation_data=(valX, [valY, valDom]),

                                callbacks=[modelcheckpnt, log_metrics(valX, [valY, valDom], pat_val, patlogDirectory,
                                                                      global_epoch_counter),
                                           csv_logger, tensbd, lrate],
                                )



        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            results_log(results_file=results_file, log_dir=log_dir, log_name=log_name, params=params)

        y_pred = model.predict(valX)[1]
        loss = K.eval(K.mean(K.variable(K.eval(keras.loss.catagorical_crossentropy(K.variable(valDom), K.variable(y_pred))))))
        loss = -loss
        print(params)
        # params['']
        return loss

    #from skopt.space import *
    from hyperopt import hp, Trials, fmin, tpe
    trials = Trials()
    best = fmin(objective,
                space=[hp.uniform('_hp_lambda', 0.001, 10),
                hp.choice('_dropout', [0.35, 0.4, 0.45, 0.5]),
                hp.uniform('_lr', 0.0001, .005)],
                algo= tpe.suggest,
                max_evals = 20,
                trials = trials
                )

    print(best)