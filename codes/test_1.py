##test
import numpy as np
import copy
from datetime import date
from scipy.io import loadmat, savemat, whosmat
from keras import initializers
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Activation, add, Dropout, merge
from keras.optimizers import Nadam
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras.utils import to_categorical#, plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau, LearningRateScheduler, Callback
import os
from keras import backend as K
from keras.constraints import max_norm

from sklearn.metrics import log_loss, accuracy_score
import sys
from sklearn.model_selection import train_test_split
sys.setrecursionlimit(3000)

class Scale(Layer):
    '''Custom Layer for ResNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='he_normal', gamma_init='he_normal', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def compute_weight(Y,classes):
		num_samples=len(Y)
		n_classes=len(classes)
		num_bin=np.bincount(Y[:,0])
		class_weights={i:(num_samples/(n_classes*num_bin[i])) for i in range(6)}
		return class_weights


def res_subsam(input_tensor,filters,kernel_size,subsam):
	eps= 1.1e-5
	nb_filter1, nb_filter2 = filters
	x = BatchNormalization(epsilon=eps, axis=-1)(input_tensor)
	x = Scale(axis=-1)(x)
	x = Activation('relu')(x)
	x = Dropout(rate=dropout_rate,seed=1)(x)
	x = Conv1D(filters=nb_filter1,kernel_initializer=initializers.he_normal(seed=1),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(x) ##
	x = MaxPooling1D(pool_size=subsam)(x)
	x = BatchNormalization(epsilon=eps, axis=-1)(x)
	x = Scale(axis=-1)(x)
	x = Activation('relu')(x)
	x = Dropout(rate=dropout_rate,seed=1)(x)
	x = Conv1D(filters=nb_filter2,kernel_initializer=initializers.he_normal(seed=1),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(x) ##	
	short = Conv1D(filters=nb_filter2,kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm),kernel_initializer=initializers.he_normal(seed=1))(input_tensor) ##
	short = MaxPooling1D(pool_size=subsam)(short)
	x = add([x,short])
	return x
	
def res_nosub(input_tensor,filters,kernel_size):
	eps= 1.1e-5
	nb_filter1, nb_filter2 = filters
	x = BatchNormalization(epsilon=eps, axis=-1)(input_tensor)
	x = Scale(axis=-1)(x)
	x = Activation('relu')(x)
	x = Dropout(rate=dropout_rate,seed=1)(x)
	x = Conv1D(filters=nb_filter1,kernel_initializer=initializers.he_normal(seed=1),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(x) ##
	x = BatchNormalization(epsilon=eps, axis=-1)(x)
	x = Scale(axis=-1)(x)
	x = Activation('relu')(x)
	x = Dropout(rate=dropout_rate,seed=1)(x)
	x = Conv1D(filters=nb_filter2,kernel_initializer=initializers.he_normal(seed=1),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(x) ##	
	x = add([x,input_tensor])
	return x
	
def res_first(input_tensor,filters,kernel_size):
	eps=1.1e-5
	nb_filter1, nb_filter2 = filters
	x = Conv1D(filters=nb_filter1,kernel_initializer=initializers.he_normal(seed=1),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(input_tensor) ##
	x = BatchNormalization(epsilon=eps, axis=-1)(x)
	x = Scale(axis=-1)(x)
	x = Activation('relu')(x)
	x = Dropout(rate=dropout_rate,seed=1)(x)
	x = Conv1D(filters=nb_filter2,kernel_initializer=initializers.he_normal(seed=1),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(x) ##	
	x = add([x,input_tensor])
	return x
	
	
def irfanet(eeg_length,num_classes, kernel_size, load_path):
	eps = 1.1e-5
	
	EEG_input = Input(shape=(eeg_length,1))
	x = Conv1D(filters=64,kernel_size=kernel_size,kernel_initializer=initializers.he_normal(seed=1),padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(EEG_input) ##
	x = BatchNormalization(epsilon=eps, axis=-1)(x)
	x = Scale(axis=-1)(x)
	x = Activation('relu')(x)
	
	x = res_first(x,filters=[64,64],kernel_size=kernel_size)
	x = res_subsam(x,filters=[64,64],kernel_size=kernel_size,subsam=2)
	x = res_nosub(x,filters=[64,64],kernel_size=kernel_size)
	x = res_subsam(x,filters=[64,128],kernel_size=kernel_size,subsam=2)
	x = res_nosub(x,filters=[128,128],kernel_size=kernel_size)
	x = res_subsam(x,filters=[128,128],kernel_size=kernel_size,subsam=2)
	x = res_nosub(x,filters=[128,128],kernel_size=kernel_size)
	x = res_subsam(x,filters=[128,192],kernel_size=kernel_size,subsam=2)
	x = res_nosub(x,filters=[192,192],kernel_size=kernel_size)
	x = res_subsam(x,filters=[192,192],kernel_size=kernel_size,subsam=2)
	x = res_nosub(x,filters=[192,192],kernel_size=kernel_size)
	x = res_subsam(x,filters=[192,256],kernel_size=kernel_size,subsam=2)
	x = res_nosub(x,filters=[256,256],kernel_size=kernel_size)
	x = res_subsam(x,filters=[256,256],kernel_size=kernel_size,subsam=2)
	x = res_nosub(x,filters=[256,256],kernel_size=kernel_size)
	x = res_subsam(x,filters=[256,512],kernel_size=kernel_size,subsam=2)
	x = BatchNormalization(epsilon=eps, axis=-1)(x)
	x = Scale(axis=-1)(x)
	x = Activation('relu')(x)
	x = Flatten()(x)
	x = Dense(num_classes,activation='softmax',kernel_initializer=initializers.he_normal(seed=1),kernel_constraint=max_norm(maxnorm),use_bias=bias)(x) ##
	
	model = Model(EEG_input, x)
	model.load_weights(filepath=load_path,by_name=False) ### LOAD WEIGHTS
	adm = Nadam(lr=lr)
	model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def lr_schedule(epoch):
	if epoch<=5:
		lr_rate=1e-3
	else:
		lr_rate=1e-4-epoch*1e-8
	return lr_rate

class show_lr(Callback):
		def on_epoch_begin(self, epoch, logs):
			print('Learning rate:')
			print(float(K.get_value(self.model.optimizer.lr)))

		
if __name__ == '__main__':
	
############################################################################################################################
#### INITIALIZE!!

	num_classes = 6
	batch_size = 8 #8
	epochs = 200
	file_name = 'eog_rk_new_notrans_234rejects_relabeled.mat'
	eeg_length = 3000
	kernel_size=16
	save_dir = os.path.join(os.getcwd(),'saved_models_keras') #os.getcwd() Return a string representing the current working directory
	model_name = 'keras_1Dconvnet_eog_trained_model.h5'
	bias=False
	maxnorm=4.
	load_path='/home/prio/Keras/thesis/irfanet-34/tmp/2017-10-29/4weights.20-0.8196.hdf5'
	run_idx=5
	dropout_rate=0.2
	initial_epoch=21
	lr=1e-5
	lr_decay=1e-8
	lr_reduce_factor=0.5 
	patience=4 #for reduceLR
	cooldown=0 #for reduceLR
	
	
#############################################################################################################################
	
	#use scipy.io to convert .mat to numpy array
	mat_cont = loadmat(file_name)
	x_test = mat_cont['dat']
	y__test = mat_cont['hyp']
	y__test=y__test-1
	
	y_test = to_categorical(y__test, num_classes)
	print('y_train shape:', y_train.shape)
	
	model = irfanet(eeg_length=eeg_length,num_classes=num_classes, kernel_size=kernel_size, load_path=load_path)
	#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB')

	pred= model.predict(x_test,batch_size=batch_size, verbose=1)
	print(pred)
	
	score = log_loss(y__test,pred)
	score_ = accuracy_score(y__test,pred)
	print(score)
	print(score_)

	
	
		
	
