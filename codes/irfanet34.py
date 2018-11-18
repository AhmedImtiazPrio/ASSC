##Class for irfanet -34

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Activation, add, Dropout, merge
from keras.optimizers import Nadam
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec

class irfanet34:
	
	def __init__(self,eeg_length,num_classes, kernel_size, load_path, lr, seed=1, bias, **kwargs):
		self.seed=seed
		self.eeg_length=eeg_length
		self.num_classes=num_classes
		self.kernel_size=kernel_size
		self.load_path=load_path
		self.lr=lr
		self.bias=bias
		self.dropout_rate=dropout_rate
		
	
	def res_subsam(input_tensor,filters,kernel_size,subsam,bias):
		eps= 1.1e-5
		nb_filter1, nb_filter2 = filters
		x = BatchNormalization(epsilon=eps, axis=-1)(input_tensor)
		x = Scale(axis=-1)(x)
		x = Activation('relu')(x)
		x = Dropout(rate=dropout_rate,seed=self.seed)(x)
		x = Conv1D(filters=nb_filter1,kernel_initializer=initializers.he_normal(seed=self.seed),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(x) ##
		x = MaxPooling1D(pool_size=subsam)(x)
		x = BatchNormalization(epsilon=eps, axis=-1)(x)
		x = Scale(axis=-1)(x)
		x = Activation('relu')(x)
		x = Dropout(rate=dropout_rate,seed=self.seed)(x)
		x = Conv1D(filters=nb_filter2,kernel_initializer=initializers.he_normal(seed=self.seed),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(x) ##	
		short = Conv1D(filters=nb_filter2,kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm),kernel_initializer=initializers.he_normal(seed=self.seed))(input_tensor) ##
		short = MaxPooling1D(pool_size=subsam)(short)
		x = add([x,short])
		return x
		
	def res_nosub(input_tensor,filters,kernel_size,bias):
		eps= 1.1e-5
		nb_filter1, nb_filter2 = filters
		x = BatchNormalization(epsilon=eps, axis=-1)(input_tensor)
		x = Scale(axis=-1)(x)
		x = Activation('relu')(x)
		x = Dropout(rate=dropout_rate,seed=self.seed)(x)
		x = Conv1D(filters=nb_filter1,kernel_initializer=initializers.he_normal(seed=self.seed),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(x) ##
		x = BatchNormalization(epsilon=eps, axis=-1)(x)
		x = Scale(axis=-1)(x)
		x = Activation('relu')(x)
		x = Dropout(rate=dropout_rate,seed=self.seed)(x)
		x = Conv1D(filters=nb_filter2,kernel_initializer=initializers.he_normal(seed=self.seed),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(x) ##	
		x = add([x,input_tensor])
		return x
		
	def res_first(input_tensor,filters,kernel_size,bias):
		eps=1.1e-5
		nb_filter1, nb_filter2 = filters
		x = Conv1D(filters=nb_filter1,kernel_initializer=initializers.he_normal(seed=self.seed),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(input_tensor) ##
		x = BatchNormalization(epsilon=eps, axis=-1)(x)
		x = Scale(axis=-1)(x)
		x = Activation('relu')(x)
		x = Dropout(rate=dropout_rate,seed=self.seed)(x)
		x = Conv1D(filters=nb_filter2,kernel_initializer=initializers.he_normal(seed=self.seed),kernel_size=kernel_size,padding='same',use_bias=bias,kernel_constraint=max_norm(maxnorm))(x) ##	
		x = add([x,input_tensor])
		return x
		
		
	def irfanet(self):
		eps = 1.1e-5
		
		EEG_input = Input(shape=(self.eeg_length,1))
		x = Conv1D(filters=64,kernel_size=self.kernel_size,kernel_initializer=initializers.he_normal(seed=self.seed),padding='same',use_bias=self.bias,kernel_constraint=max_norm(maxnorm))(EEG_input) ##
		x = BatchNormalization(epsilon=eps, axis=-1)(x)
		x = Scale(axis=-1)(x)
		x = Activation('relu')(x)
		
		x = res_first(x,filters=[64,64],kernel_size=self.kernel_size,bias=self.bias)
		x = res_subsam(x,filters=[64,64],kernel_size=self.kernel_size,subsam=2,bias=self.bias)
		x = res_nosub(x,filters=[64,64],kernel_size=self.kernel_size,bias=self.bias)
		x = res_subsam(x,filters=[64,128],kernel_size=self.kernel_size,subsam=2,bias=self.bias)
		x = res_nosub(x,filters=[128,128],kernel_size=self.kernel_size,bias=self.bias)
		x = res_subsam(x,filters=[128,128],kernel_size=self.kernel_size,subsam=2,bias=self.bias)
		x = res_nosub(x,filters=[128,128],kernel_size=self.kernel_size,bias=self.bias)
		x = res_subsam(x,filters=[128,192],kernel_size=self.kernel_size,subsam=2,bias=self.bias)
		x = res_nosub(x,filters=[192,192],kernel_size=self.kernel_size,bias=self.bias)
		x = res_subsam(x,filters=[192,192],kernel_size=self.kernel_size,subsam=2,bias=self.bias)
		x = res_nosub(x,filters=[192,192],kernel_size=self.kernel_size,bias=self.bias)
		x = res_subsam(x,filters=[192,256],kernel_size=self.kernel_size,subsam=2,bias=self.bias)
		x = res_nosub(x,filters=[256,256],kernel_size=self.kernel_size,bias=self.bias)
		x = res_subsam(x,filters=[256,256],kernel_size=self.kernel_size,subsam=2,bias=self.bias)
		x = res_nosub(x,filters=[256,256],kernel_size=self.kernel_size,bias=self.bias)
		x = res_subsam(x,filters=[256,512],kernel_size=self.kernel_size,subsam=2,bias=self.bias)
		x = BatchNormalization(epsilon=eps, axis=-1)(x)
		x = Scale(axis=-1)(x)
		x = Activation('relu')(x)
		x = Flatten()(x)
		x = Dense(self.num_classes,activation='softmax',kernel_initializer=initializers.he_normal(seed=self.seed),kernel_constraint=max_norm(maxnorm),use_bias=self.bias)(x) ##
		
		model = Model(EEG_input, x)
		model.load_weights(filepath=load_path,by_name=False) ### LOAD WEIGHTS
		adm = Nadam(lr=self.lr)
		model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
