from __future__ import print_function, absolute_import, division
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from datetime import datetime
import argparse
import os
import tables
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, plot_model
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

from modules import *
from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("fold",
                        help="matfile to use")
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

    model_dir = os.path.join(os.getcwd(),'..','models')
    fold_dir = os.path.join(os.getcwd(),'..','data')
    log_dir = os.path.join(os.getcwd(),'..','logs')
    log_name = foldname + ' ' + str(datetime.now())
    if not os.path.exists(model_dir + log_name):
        os.makedirs(model_dir + log_name)
    checkpoint_name = os.path.join(model_dir,log_name,'weights.{epoch:04d}-{val_acc:.4f}.hdf5') # make sure separate
                                                                                        # folder for each log_name
    results_file = os.path.join(os.getcwd(), '..', 'results.csv')

    params={               # still not universal

        'num_classes':6,  ### automate; number of classes depends on data fold
        'batch_size':batch_size,
        'epochs':epochs,
        'foldname':foldname,
        'random_seed':random_seed,
        'load_path':load_path,
        'shuffle':True,
        'initial_epoch':initial_epoch,
        'verbose':verbose,
        'eeg_length':3000,
        'kernel_size':16,
        'bias':True,
        'maxnorm':4.,
        'dropout_rate':0.5,
        'dropout_rate_dense':0.,
        'padding':'valid',
        'activation_function':'relu',
        'subsam':2,
        'trainable':True,
        'lr':.0001,
        'lr_decay':1e-5,

    }

    ########### Data Prep ################

    mat_cont = tables.open_file(os.path.join(fold_dir,foldname+'.mat'))
    X = mat_cont['data']
    Y = mat_cont['hyp']
    patID = mat_cont['patID']
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, random_state=random_seed)
    trainY = to_categorical(trainY, params['num_classes'])
    valY = to_categorical(valY, params['num_classes'])

    ########### Create Model ################

    top_model = eegnet(**params)  # might have bugs; sub modules need kwargs integration
    x = Flatten()(top_model)
    x = Dense(params['num_classes'], activation='softmax', kernel_initializer=initializers.he_normal(seed=random_seed),
              kernel_constraint=max_norm(params['maxnorm']), use_bias=True)(x)  ##

    model = Model(top_model.input, x)
    model.summary()
    if load_path:  # If path for loading model was specified
        model.load_weights(filepath=load_path, by_name=False)
    plot_model(model, to_file='model.png', show_shapes=True)
    model_json = model.to_json()
    with open(os.path.join(model_dir,log_name,'model.json'), "w") as json_file:
        json_file.write(model_json)
    adm = Adam(**params)  # might have bugs
    model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])

    ####### Define Callbacks #######

    modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                    monitor='val_acc', save_best_only=False, mode='max')
    tensbd = TensorBoard(log_dir=os.path.join(log_dir,log_name),
                         batch_size=batch_size, histogram_freq=3,
                         write_grads=True,
                         # embeddings_freq=99,
                         # embeddings_layer_names=embedding_layer_names,
                         # embeddings_data=x_val,
                         # embeddings_metadata=metadata_file,
                         write_images=False)
    csv_logger = CSVLogger(os.path.join(log_dir,log_name,'training.csv'))

    if args.classweights:
        params['class_weight'] = compute_weight(trainY, np.unique(trainY))
    else:
        params['class_weight'] = dict(zip(np.r_[0:params['num_classes']],np.ones(params['num_classes']))) # weighted 1

    ####### Train #######

    try:

        model.fit(trainX, trainY,

                verbose=2,
                validation_data=(valX, valY),
                callbacks=[modelcheckpnt,
                         log_metrics(valX, valY, patID),
                         tensbd, csv_logger],
                  **params)  # might have bugs
        results_log(results_file,params)

    except KeyboardInterrupt:

        results_log(results_file, params)
