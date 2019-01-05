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
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import pandas as pd

from modules import *
from utils import *

if __name__ == '__main__':
	
	
	
	############ পারসার এই ক্যাচাল এইখান থেকে শুরু #######################
	
	
	########## পারসার ডেফিনিশন ########
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

	args = parser.parse_args()
	print("%s selected" % (args.fold))
	foldname = args.fold

	###### পারসার থেকে নিয়ে ভ্যারিয়েবল গুলাতে মান বসানো, মান না থাকলে ডিফল্ট কত হবে সেইগুলাও বসান ########
	
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
	
	
	######## এই পর্যন্ত শুধু পারসার থেকে মান নিয়ে ভ্যারিয়েবল গুলা তে বসেছে। ###############
	
	### ডিরেক্টরি ডিফাইন করা জেনারেলাইজ করে ####

	# model_dir = os.path.join(os.getcwd(),'..','models').replace('\\','/')
	# fold_dir = os.path.join(os.getcwd(),'..','data').replace('\\','/')
	# log_dir = os.path.join(os.getcwd(),'..','logs').replace('\\','/')
	# #log_name= os.path.join(foldname,'_',str(datetime.now()) ).replace('\\','/')
	# log_name = (foldname + '_' + str(datetime.now())).replace(' ','').replace('\\','/')
	# print(os.path.join(model_dir,log_name))
	# if not os.path.exists(os.path.join(model_dir,log_name)):
	# 	new_dir =(os.path.join(os.getcwd(),'..','models',log_name)).replace('\\','/').replace(':','')
	# 	print(new_dir)
	# 	os.makedirs(new_dir)
	# checkpoint_name = os.path.join(model_dir,log_name,'weights.{epoch04d}-{val_acc.4f}.hdf5').replace(':','') # make sure separate
	# 																					# folder for each log_name
	# results_file = os.path.join(os.getcwd(), '..', 'results.csv').replace('\\','/')
	#

	model_dir = 'E:/SleepWell/ASSC-master/models'
	fold_dir = 'E:/SleepWell/ASSC-master/data'
	log_dir = 'E:/SleepWell/ASSC-master/logs'
	log_name = foldname + str(datetime.now())
	if not os.path.exists(os.path.join(model_dir,log_name)):
		new_dir =(os.path.join(model_dir,log_name))
		print(new_dir)
		os.makedirs(new_dir)
	checkpoint_name = os.path.join(model_dir,log_name,'weights.{epoch04d}-{val_acc.4f}.hdf5')
	results_file = os.path.join('E:/SleepWell/ASSC-master/','results.csv')
	##### ডিরেক্টরি ক্যাচাল শেষ #####

	##### প্যারামস হচ্ছে একটা লিস্ট যেটা নেটওয়ার্ক কে খাওয়াতে হবে। এই লিস্টে সব হাইপারপ্যারামিটার থেকে শুরু করে ফোল্ডারের নাম সব থাকবে। এটা শিখলাম। কাজ করানো টা শিখতে হবে ################
	
	params={               # still not universal

		'num_classes':7,  ### automate; number of classes depends on data fold
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

	# mat_cont = tables.open_file(os.path.join(fold_dir,foldname))

	#Elegant one
	#df2 = pd.read_csv( (os.path.join(fold_dir,foldname),header=None)
	#কামলা কাউন্টারপার্ট
	df2 = pd.read_csv('E:/SleepWell/ASSC-master/data/purifiedallDataChannel2.csv',header=None)
	df2.rename({3000: 'hyp', 3001: 'epoch', 3002: 'patID'}, axis="columns", inplace=True)


	#X = mat_cont['data']
	#Y = mat_cont['hyp']
	#patID = mat_cont['patID']

	
	#trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, random_state=random_seed) # patID split koro bacha
	trainX, valX, trainY, valY, pat_train, pat_val = patientSplitter('randomizedIDs.csv', df2, 0.7)
	print("SPLITTING IS DONE")
	print("the shape of train Y is ")
	#####এই স্প্লিটিং করা যাবে না। লিখতে হবে। পেশেন্ট আইডি এর উপর বেইজ করে স্প্লিট করতে হবে। 
	
	
	######পেশেন্ট আইডী বেজ করে স্প্লিট করা নিয়ে কাজ করতে হবে এইখানে। ১৫ জনের ডাটা যাবে ট্রেনিং এ, বাকি দের ডেটা যাবে ভ্যালিডেশনে। ##############
	trainY = to_categorical(trainY, params['num_classes'])
	valY = to_categorical(valY, params['num_classes'])

	########### Create Model ################
	###### মডেলের শুরুর দিকের লেয়ার গুলা মডুলস নামের পাই ফাইলে আছে। এখানে শুধু ডেন্স লেয়ার গুলা আলাদা করে জইন করা হচ্ছে, কারণ এইগুলা তেই চেঞ্জ আসবে।  ######
	##### হাইপারপ্যারামিটার গুলা ট্রেইন করার জন্য শুধু হাইপার প্যারামিটার গুলা সম্ভলিত অংশ আলাদা করে লেখা হচ্ছে। ##################
	#শিখলাম ব্যপারটা। #
	
	# প্রথমে ইইজি নেট টা কে তৈরি করা, সব প্যারামিটার তাকে বলে দিয়ে ###
	top_model = eegnet(**params)  # might have bugs; sub modules need kwargs integration
	#    top_model = eegnet(params)

	# এর পরে সেটার আউটপুট কে ফ্ল্যাটেন করা, যাতে করে ডেন্স লেয়ার এড করা যায়।  ফ্ল্যাটেন করার পরে প্রথম ডেন্স লেয়ার এড করা। আরো বেশি ডেন্স লেয়ার এড করা যেতে পারে ব্যপার টা তে। পরে চেষ্টা করে দেখতে হবে বিভিন্ন কনফিগারেশন।   ####
	#এখানে যেমন প্রথম ডেন্স লেয়ারের সাথেই সফটম্যাক্স করে আঊটপুট দেওয়া। এমন না করে আরেকটা ডেন্স লেয়ার রেখে সেইটা তে সফট্ম্যাএক্স করা উচিত। রান দেওার পরে সেই কাজ করতে হবে ###
	x = Flatten()(top_model)
	x = Dense(params['num_classes'], activation='softmax', kernel_initializer=initializers.he_normal(seed=random_seed),
			  kernel_constraint=max_norm(params['maxnorm']), use_bias=True)(x)  ##

	#model = Model(top_model.Input, x) # এখানে দুইটা মডেল জোড়া লেগে যাচ্ছে। টপ মডেল, আর পরের ডেন্স করার পরের অংশ - এই দুইটা।
	model = Model(inputs = top_model.inputs, outputs = x)
	model.summary() # মডেলের সামারি
	if load_path:  # If path for loading model was specified
		model.load_weights(filepath=load_path, by_name=False)
	plot_model(model, to_file='model.png', show_shapes=True) #  মডেল কে ইমেজ ফাইলে আঁকা
	model_json = model.to_json() #জেসন ফাইলে লেখা হচ্ছে মডেল টা কে। সব ধরনের প্রিকশন নিয়ে রাখা, আর কি।
	with open(os.path.join(model_dir,log_name,'model.json').replace('\\','/'), "w") as json_file:
		json_file.write(model_json)
	model.compile(optimizer=opt(**params), loss='categorical_crossentropy', metrics=['accuracy']) # মডেল কম্পাইলেশন। টেক্সটবুক আচরণ, অবশেষে

	####### Define Callbacks #######
	
	### ভ্যালিডেশন একুরেসির উপর বেজ করে চেজপয়েন্ট নিয়ে রাখা মডেল সেভ করার জন্য #########
	modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
									monitor='val_acc', save_best_only=False, mode='max')
	### টেন্সরবোরড ইন্সট্যান্স কল করা ######								
	tensbd = TensorBoard(log_dir=os.path.join(log_dir,log_name),
						 batch_size=batch_size, histogram_freq=3,
						 write_grads=True,
						 # embeddings_freq=99,
						 # embeddings_layer_names=embedding_layer_names,
						 # embeddings_data=x_val,
						 # embeddings_metadata=metadata_file,
						 write_images=False)
						 
	##### সিএসভি লগারের ইন্সট্যান্স তৈরি করা, লগ সেইভ করার জন্য ###########					 
	csv_logger = CSVLogger(os.path.join(log_dir,log_name,'training.csv').replace('\\','/'))

	
	
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
