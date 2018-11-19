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
import pandas as pd
import tables

def irfanet(eeg_length, num_classes, kernel_size, load_path):
    eps = 1.1e-5

    EEG_input = Input(shape=(eeg_length, 1))
    x = Conv1D(filters=64, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=1), padding='same',
               use_bias=bias, kernel_constraint=max_norm(maxnorm))(EEG_input)  ##
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Scale(axis=-1)(x)
    x = Activation('relu')(x)

    x = res_first(x, filters=[64, 64], kernel_size=kernel_size)
    x = res_subsam(x, filters=[64, 64], kernel_size=kernel_size, subsam=2)
    x = res_nosub(x, filters=[64, 64], kernel_size=kernel_size)
    x = res_subsam(x, filters=[64, 128], kernel_size=kernel_size, subsam=2)
    x = res_nosub(x, filters=[128, 128], kernel_size=kernel_size)
    x = res_subsam(x, filters=[128, 128], kernel_size=kernel_size, subsam=2)
    x = res_nosub(x, filters=[128, 128], kernel_size=kernel_size)
    x = res_subsam(x, filters=[128, 192], kernel_size=kernel_size, subsam=2)
    x = res_nosub(x, filters=[192, 192], kernel_size=kernel_size)
    x = res_subsam(x, filters=[192, 192], kernel_size=kernel_size, subsam=2)
    x = res_nosub(x, filters=[192, 192], kernel_size=kernel_size)
    x = res_subsam(x, filters=[192, 256], kernel_size=kernel_size, subsam=2)
    x = res_nosub(x, filters=[256, 256], kernel_size=kernel_size)
    x = res_subsam(x, filters=[256, 256], kernel_size=kernel_size, subsam=2)
    x = res_nosub(x, filters=[256, 256], kernel_size=kernel_size)
    x = res_subsam(x, filters=[256, 512], kernel_size=kernel_size, subsam=2)
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Scale(axis=-1)(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', kernel_initializer=initializers.he_normal(seed=1),
              kernel_constraint=max_norm(maxnorm), use_bias=bias)(x)  ##

    model = Model(EEG_input, x)
    # model.load_weights(filepath=load_path,by_name=False) ### LOAD WEIGHTS
    adm = Adam(lr=lr, decay=lr_decay)
    model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':

    ############################################################################################################################
    #### INITIALIZE!!

    num_classes = 6
    batch_size = 32  # 8
    epochs = 200
    file_name = 'eog_rk_new_notrans_234rejects_relabeled.mat'
    eeg_length = 3000
    kernel_size = 16
    save_dir = os.path.join(os.getcwd(), 'saved_models_keras')
    model_name = 'keras_1Dconvnet_eog_trained_model.h5'
    bias = True
    maxnorm = 4.
    # load_path='/home/prio/Keras/thesis/irfanet-34/tmp/2017-10-29/4weights.20-0.8196.hdf5'
    load_path = None
    run_idx = 3
    dropout_rate = 0.2
    initial_epoch = 0
    lr = .0001
    lr_decay = 1e-5
    lr_reduce_factor = 0.5
    patience = 4  # for reduceLR
    cooldown = 0  # for reduceLR

    #############################################################################################################################

    # use scipy.io to convert .mat to numpy array
    mat_cont = loadmat(file_name)
    X = mat_cont['dat']
    Y = mat_cont['hyp']
    Y = Y - 1

    # Use random splitting into training and test
    x_train, x_test, y__train, y__test = train_test_split(X, Y, test_size=0.2, random_state=1)
    x_train = np.reshape(x_train, (x_train.shape[0], 3000, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 3000, 1))

    # Use alternate epochs
    # A = np.reshape(X,(47237,3000,1))
    # x_test = A[::2,:,:]
    # y_test = Y[::2]
    # x_train = A[2:X.shape[0]:2,:,:]
    # y_train = Y[2:Y.shape[0]:2]

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('Training Distribution')
    print(np.bincount(y__train[:, 0]))
    print('Testing Distribution')
    print(np.bincount(y__test[:, 0]))

    y_train = to_categorical(y__train, num_classes)
    y_test = to_categorical(y__test, num_classes)
    print('y_train shape:', y_train.shape)

    x = irfanet(eeg_length=eeg_length, num_classes=num_classes, kernel_size=kernel_size, load_path=load_path)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', kernel_initializer=initializers.he_normal(seed=1),
              kernel_constraint=max_norm(maxnorm), use_bias=bias)(x)  ##

    model = Model(EEG_input, x)
    # model.load_weights(filepath=load_path,by_name=False) ### LOAD WEIGHTS
    adm = Adam(lr=lr, decay=lr_decay)
    model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])

    # setting up checkpoint save directory/ log names
    checkpoint_path = os.path.join(os.path.join(os.getcwd(), 'tmp'), str(date.today()))
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_name = checkpoint_path + '/' + str(run_idx) + 'weights.{epoch:02d}-{val_acc:.4f}.hdf5'
    log_name = 'logs' + str(date.today()) + '_' + str(run_idx)

    # Callbacks
    mdlchk = ModelCheckpoint(filepath=checkpoint_name, monitor='val_acc', save_best_only=False, mode='max')
    tensbd = TensorBoard(log_dir='./logs/' + log_name, batch_size=batch_size, write_images=True)
    csv_logger = CSVLogger('./logs/training_' + log_name + '.log', separator=',', append=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_reduce_factor, patience=patience, min_lr=0.00001,
                                  verbose=1, cooldown=cooldown)
    lr_ = LearningRateScheduler(lr_schedule)
    lr_print = show_lr()

    # class_weight={0:3.3359,1:0.3368,2:3.0813,3:2.7868,4:0.7300,5:1.4757}
    class_weight = compute_weight(y__train, np.unique(y__train))
    print(class_weight)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=2,
              validation_data=(x_test, y_test),
              callbacks=[mdlchk, tensbd, csv_logger, lr_print],  # reduce_lr],
              initial_epoch=initial_epoch
              )
    # class_weight=class_weight
    # )

    pred = model.predict(x_test, batch_size=batch_size, verbose=1)
    print(pred)

    score = log_loss(y__test, pred)
    score_ = accuracy_score(y__test, pred)
    print(score)
    print(score_)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
