from keras.layers import Conv1D, MaxPooling1D, Activation, add, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras.engine import Layer, InputSpec
from keras.models import Model
#from keras.engine import input_layer
from keras import backend as K
from keras.constraints import max_norm
#from input_layer import Input
import tensorflow as tf
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

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='he_normal', gamma_init='he_normal', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma' % self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta' % self.name)
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


def res_subsam(input_tensor, filters=(64,64), kernel_size=16, subsam=2, dropout_rate=0.2, bias=False, maxnorm=4., **kwargs):
    eps = 1.1e-5
    nb_filter1, nb_filter2 = filters
    x = BatchNormalization(epsilon=eps, axis=-1)(input_tensor)
    x = Scale(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=dropout_rate, seed=1)(x)
    x = Conv1D(filters=nb_filter1, kernel_initializer=initializers.he_normal(seed=1), kernel_size=kernel_size,
               padding='same', use_bias=bias, kernel_constraint=max_norm(maxnorm))(x)  ##
    x = MaxPooling1D(pool_size=subsam)(x)
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Scale(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=dropout_rate, seed=1)(x)
    x = Conv1D(filters=nb_filter2, kernel_initializer=initializers.he_normal(seed=1), kernel_size=kernel_size,
               padding='same', use_bias=bias, kernel_constraint=max_norm(maxnorm))(x)  ##
    short = Conv1D(filters=nb_filter2, kernel_size=kernel_size, padding='same', use_bias=bias,
                   kernel_constraint=max_norm(maxnorm), kernel_initializer=initializers.he_normal(seed=1))(
        input_tensor)  ##
    short = MaxPooling1D(pool_size=subsam)(short)
    x = add([x, short])
    return x


def res_nosub(input_tensor, filters=(64,64), kernel_size=16, dropout_rate=0.2, bias=False, maxnorm=4., **kwargs):
    eps = 1.1e-5
    nb_filter1, nb_filter2 = filters
    x = BatchNormalization(epsilon=eps, axis=-1)(input_tensor)
    x = Scale(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=dropout_rate, seed=1)(x)
    x = Conv1D(filters=nb_filter1, kernel_initializer=initializers.he_normal(seed=1), kernel_size=kernel_size,
               padding='same', use_bias=bias, kernel_constraint=max_norm(maxnorm))(x)  ##
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Scale(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=dropout_rate, seed=1)(x)
    x = Conv1D(filters=nb_filter2, kernel_initializer=initializers.he_normal(seed=1), kernel_size=kernel_size,
               padding='same', use_bias=bias, kernel_constraint=max_norm(maxnorm))(x)  ##
    x = add([x, input_tensor])
    return x


def res_first(input_tensor, filters=(64,64), kernel_size=16, dropout_rate=0.2, bias=False, maxnorm=4., **kwargs):
    eps = 1.1e-5
    nb_filter1, nb_filter2 = filters
    x = Conv1D(filters=nb_filter1, kernel_initializer=initializers.he_normal(seed=1), kernel_size=kernel_size,
               padding='same', use_bias=bias, kernel_constraint=max_norm(maxnorm))(input_tensor)  ##
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Scale(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=dropout_rate, seed=1)(x)
    x = Conv1D(filters=nb_filter2, kernel_initializer=initializers.he_normal(seed=1), kernel_size=kernel_size,
               padding='same', use_bias=bias, kernel_constraint=max_norm(maxnorm))(x)  ##
    x = add([x, input_tensor])
    return x


def eegnet(eeg_length=3000, kernel_size=16, bias=False, maxnorm=4., **kwargs):

    '''
    Top model for the CNN
    Add details of module in docstring
        '''

    eps = 1.1e-5

    #inputs = K.placeholder(shape=(batch_size, eeg_length,1))
    #x = Input(dtype= 'float32', shape=(eeg_length,1))
    EEG_input = Input(shape=(eeg_length,1))
    x = Conv1D(filters=64, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=1), padding='same',
               use_bias=bias, kernel_constraint=max_norm(maxnorm))(EEG_input)  ##
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Scale(axis=-1)(x)
    x = Activation('relu')(x)  # ব্যাচ নরমালাইজ করা ভার্শন টা কে নিয়ে স্কেলের মধ্যে ঢুকানো 

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
    x = Model(EEG_input,x)
    # tf.keras.backend.eval(x)
    return x
