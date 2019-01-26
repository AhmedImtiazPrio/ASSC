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
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, Callback
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from modules import *
#from utils import *



def standardnormalization(distribution):
    from sklearn.preprocessing import StandardScaler
    data = distribution.flatten('A')
    data = np.expand_dims(data, axis=1)
    scaler = StandardScaler()
    outdata = scaler.fit_transform(data)
    toutdata = outdata.reshape(int(outdata.shape[0]/3000),3000)
    return toutdata

def compute_weight(Y, classes):
    num_samples = len(Y)
    n_classes = len(classes)
    Y = Y.astype(int)
    Y = np.expand_dims(Y, axis=1)
    num_bin = np.bincount(Y[:, 0])
    class_weights = {i: (num_samples / (n_classes * num_bin[i])) for i in range(6)}
    return class_weights

def patientSplitter(randomIDfile,df2,split_portion, totalPat = 61):
    import pandas as pd

    df1 = pd.read_csv(randomIDfile,header=None)
    split_portion_numer=int(split_portion*totalPat)

    train_pat_list = [int(each) for each in df1.iloc[:split_portion_numer].values]
    test_pat_list = [int(each) for each in df1.iloc[split_portion_numer:].values]
    print(test_pat_list)
    df3 = []
    df4 = []
    for pat_ID in train_pat_list:
        df3.append(df2[df2.patID == pat_ID].values)
        print(pat_ID)
    for pat_ID in test_pat_list:
        df4.append(df2[df2.patID == pat_ID].values)
        print(pat_ID)
    del df2
    df3 = pd.np.vstack(df3)
    df4 = pd.np.vstack(df4)
    X_train = df3[:, :3000]
    X_test= df4[:, :3000]
    Y_train= df3[:,3000]
    Y_test= df4[:,3000]
    pat_train=df3[:, 3002:3003]
    pat_test= df4[:, 3002:3003]

    del pd
    return X_train,X_test,Y_train,Y_test,pat_train,pat_test

def results_log(results_file, log_dir, log_name, params):
    df = pd.read_csv(results_file)
    df1 = pd.read_csv(os.path.join(log_dir, log_name, 'training.csv').replace('\\', '/'))

    new_entry= params
    new_entry.pop('class_weight')
    new_entry.pop('lr')
    a = df1.head()
    a = a.join(pd.DataFrame(params, index=[0]))
    df2 = pd.concat([df, a], axis=0)
    df2.to_csv(results_file, index=False)
    print("saving results to csv")
    #print(params['class_weight'])







def epoch_reduction(trainX, trainY, wakeReduction=False, wakeRedSize=0.0, s1Reduction=False, s1RedSize=0.0):

    if(wakeReduction):
        wakecount = 0
        trainwakemask = trainY == 6
        for i in range(0, len(trainwakemask)):
            if trainwakemask[i] == 1:
                wakecount = wakecount + 1
        print(wakecount)
        willdeletewake = int(wakecount*wakeRedSize)

        for i in range(willdeletewake, len(trainwakemask)):
            trainwakemask[i] = 0
        trainwakemask = ~trainwakemask
        trainX = trainX[trainwakemask]
        trainY = trainY[trainwakemask]

    if (s1Reduction):
        s1count = 0
        trains1mask = trainY == 2
        for i in range(0, len(trains1mask)):
            if trains1mask[i] == 1:
                s1count = s1count + 1
        print(s1count)
        willdeletes1 = int(s1count * s1RedSize)

        for i in range(willdeletes1, len(trains1mask)):
            trains1mask[i] = 0

        trains1mask = ~trains1mask
        trainX = trainX[trains1mask]
        trainY = trainY[trains1mask]

    return trainX, trainY









class log_metrics( Callback):
    def __init__(self, valX, valY, patID, patlogDirectory, global_epoch_counter, **kwargs):
        self.valY = np.argmax(valY, axis=-1)
        self.valY = np.expand_dims(self.valY, axis=1)
        self.valX = valX
        self.patID = patID
        super(log_metrics,self).__init__(**kwargs)
        self.patlogDirectory = patlogDirectory
        self.global_epoch_counter = global_epoch_counter
    def on_epoch_end(self, epoch, logs):

        if logs is not None:

            predY = self.model.predict(self.valX, verbose=0)
            predY = np.argmax(predY, axis=-1)
            predY = np.expand_dims(predY, axis=1)
            patAcc = []
            print("printing the shape of predY")
            print(predY.shape)
            # for pat in np.unique(self.patID).astype(int):
            #     mask = self.patID == pat
            #     patAcc.append(accuracy_score(self.valY[mask], predY[mask]))
            #logs['PerPatientAccuracy'] = np.mean(patAcc)

            ##################################################################
            confMat = confusion_matrix(self.valY -1, predY)

            match = sum(predY == self.valY - 1)

            patsens = []
            patspec = []

            dummyvalY = self.valY -1

            for each in np.unique(dummyvalY).astype(int):
                each = int(each)
                patsens.append(confMat[each, each] / sum(confMat[each, :]))
                patspec.append((match - confMat[each, each]) / ((match - confMat[each, each] + sum(confMat[:, each]) - confMat[each, each])))


            acc = match / len(self.valY)

            patAcc = []
            for pat in np.unique(self.patID).astype(int):
                mask = self.patID[:, 0] == pat
                patAcc.append(accuracy_score(self.valY[mask] - 1, predY[mask]))
            patAccAvg = np.mean(patAcc)

            # perClassSens= []
            # perClassSpec = []
            # correctedtrueY = self.valY - 1
            # for each_class in np.unique(correctedtrueY):
            #     mask = correctedtrueY == each_class
            #     confMat = confusion_matrix(correctedtrueY[mask], predY[mask])
            #     perClassSens.append(confMat[each, each] / sum(confMat[each, :]))
            #     perClassSpec.append((match-confMat[each, each])/ ((match - confMat[each, each] + sum(confMat[:, each]) - confMat[each, each])))

            patsens = []
            patspec = []

            dummyvalY = self.valY - 1

            for each in np.unique(dummyvalY).astype(int):
                each = int(each)
                patsens.append(confMat[each, each] / sum(confMat[each, :]))
                patspec.append((match - confMat[each, each]) / (
                (match - confMat[each, each] + sum(confMat[:, each]) - confMat[each, each])))

            acc = match / len(self.valY)

            patAcc = []
            for pat in np.unique(self.patID).astype(int):
                mask = self.patID[:, 0] == pat
                patAcc.append(accuracy_score(self.valY[mask] - 1, predY[mask]))
            patAccAvg = np.mean(patAcc)
            print("patient accuracy is")
            print(patAccAvg)

            perClassAcc= []
            perClassSens = []
            perClassSpec = []
            correctedtrueY = self.valY
            print(correctedtrueY.shape)
            print(np.unique(correctedtrueY))
            for each_class in np.unique(correctedtrueY):
                print("fjskdjfh")
                each_class = int(each_class)
                mask = correctedtrueY == each_class
                totalcandidatesforthatclass = sum(mask)
                print(totalcandidatesforthatclass)
                match = sum(correctedtrueY[mask] == predY[mask])

                print(match)
                acc = match / totalcandidatesforthatclass
                print(acc)
                perClassAcc.append(acc)

                #     tp = match
                #     tn = sum(correctedtrueY[~mask] - sum(correctedtrueY[~mask]== each_class)
                #     fp = sum([])
                #     fn =

                #perClassSens.append(match / totalcandidatesforthatclass)
                #perClassSens.append()

                # confMat = confusion_matrix(correctedtrueY[mask], predY[mask])
                # print(confMat)
                print("done")
                # perClassSens.append(confMat[each_class, each_class] / sum(confMat[each_class, :]))
                # perClassSpec.append((match-confMat[each_class, each_class])/ ((match - confMat[each_class, each_class] + sum(confMat[:, each_class]) - confMat[each_class, each_class])))
            #print(perClassSens)
            #print(perClassSpec)








             #######################   LOGGING  #############################################
            logs['PerPatientAccuracy'] = np.mean(patAcc)
            # logs['PerPatientSensitivity'] = np.mean(patsens)
            # logs['PerPatientSpecificity'] = np.mean(patspec)
            #
            # logs['PerClassSensitivity'] = np.mean(perClassSens)
            # logs['PerClassSpecificity'] = np.mean(perClassSpec)
            #
            # logs['pat1Specificity'] = patspec[0]
            # logs['s2Specificity'] = patspec[1]
            # logs['s3Specificity'] = patspec[2]
            # logs['s4Specificity'] = patspec[3]
            # logs['REMSpecificity'] = patspec[4]
            # logs['WakeSpecificity'] = patspec[5]
            #
            # logs['s1Sensitivity'] = patsens[0]
            # logs['s2Sensitivity'] = patsens[1]
            # logs['s3Sensitivity'] = patsens[2]
            # logs['s4Sensitivity'] = patsens[3]
            # logs['REMSensitivity'] = patsens[4]
            # logs['WakeSensitivity'] = patsens[5]
            #
            #
            logs['s1Acc'] = perClassAcc[0]
            logs['s2Acc'] = perClassAcc[1]
            logs['s3Acc'] = perClassAcc[2]
            logs['s4Acc'] = perClassAcc[3]
            logs['REMAcc'] = perClassAcc[4]
            logs['WakeAcc'] = perClassAcc[5]

            logs['patientSpecificity'] = np.mean(patspec)
            logs['patientSenssitivity'] = np.mean(patsens)
            #

            #################################################################################




            np.savetxt(self.patlogDirectory + "epoch" + str(self.global_epoch_counter)+"patAcc.csv", patAcc, delimiter=",")
            np.savetxt(self.patlogDirectory+"patients.csv", self.patID, delimiter=",")

            #self.patID.to_csv(self.patlogDirectory+"patients.csv", index=False)

            self.global_epoch_counter = self.global_epoch_counter +1
            global_epoch_counter = self.global_epoch_counter
            lr = self.model.optimizer.lr
            if self.model.optimizer.initial_decay > 0:
                lr *= (1. / (1. + self.model.optimizer.decay * K.cast(self.model.optimizer.iterations, K.dtype(self.model.optimizer.decay))))
            t = K.cast(self.model.optimizer.iterations, K.floatx()) + 1
            lr_t = lr * (K.sqrt(1. - K.pow(self.model.optimizer.beta_2, t)) / (1. - K.pow(self.model.optimizer.beta_1, t)))
            logs['lr'] = np.array(float(K.get_value(lr_t)))


