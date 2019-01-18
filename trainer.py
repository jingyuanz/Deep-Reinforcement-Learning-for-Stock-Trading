#coding=utf-8
from config import Config
from data_util import DataUtil
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input
from keras.layers import GlobalMaxPooling1D, Reshape, TimeDistributed, Conv1D, Bidirectional, Concatenate, BatchNormalization, PReLU
from keras.layers import LSTM
import keras
from keras.regularizers import l2
import numpy as np
from keras.models import save_model, load_model
from keras import backend as K
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tushare as ts
from sklearn.metrics import roc_auc_score
import pickle
from datetime import date
from keras.utils import to_categorical

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

def auc(y_true, y_pred):
    # roc = roc_auc_score(y_true, y_pred)
    tf.initialize_local_variables()
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class Classifier:
    def __init__(self):
        self.config = Config()
        self.du = DataUtil()
        self.codes = self.du.sz50_code()
        
    def load_data(self):
        self.train, self.test = self.du.get_data_from_codes(self.codes, load=True, predict=False)
        self.train = np.array(self.train)
        self.test = np.array(self.test)
        self.train_X, self.train_Y, self.train_dates, self.train_changes, self.train_codes = self.train[:,0], self.train[:,1], self.train[:,2], self.train[:,3], self.train[:,4]
        self.test_X, self.test_Y, self.test_dates, self.test_changes, self.test_codes = self.test[:,0], self.test[:,1], self.test[:,2], self.test[:,3], self.test[:,4]
        self.train_X = [list(x) for x in self.train_X]
        self.test_X = [list(x) for x in self.test_X]
        self.train_X = np.asarray(self.train_X)
        self.test_X = np.asarray(self.test_X)
        print(self.train_X.shape)
        print(np.mean(self.test_Y), np.mean(self.train_Y))
        # for e in self.train_X:
        #     assert e.shape == my_shape
        # for e in self.test_X:
        #     assert e.shape == my_shape
        #
        
        
    def run_trainer(self, load=False):
        if load:
            self.model = load_model("model/{}_best.model".format(date.today()))
        #compile model
        else:
            self.model = self.build_model()
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer=keras.optimizers.Nadam(),
                           metrics=['accuracy'])
        self.model.summary()
        check = keras.callbacks.ModelCheckpoint("model/{}_best.model".format(date.today()), monitor='val_acc', verbose=1,
                                                save_best_only=True, save_weights_only=False, mode='auto', period=1)
        # self.labels = to_categorical(self.labels, 2)
        # self.test_y = to_categorical(self.test_y, 2)
        self.model.fit(self.train_X, self.train_Y,
                       batch_size=self.du.config.batch_size,
                       epochs=self.du.config.epochs,
                       verbose=1,
                       validation_data=(self.test_X, self.test_Y), callbacks=[check,roc_callback(training_data=(self.train_X, self.train_Y),validation_data=(self.test_X, self.test_Y))])

        self.model.save("model/{}_final.model".format(date.today()))
        preds = self.model.predict(self.test_X)
        preds = [pred for pred in preds]
        results = list(zip(self.test_codes, preds, self.test_Y, self.test_dates, self.test_changes))
        sorts = sorted(results, key=lambda x: x[1], reverse=True)
        for k, v, truth, d, change in sorts:
            print(k, v, truth, d, change)

    def build_model(self):
        input = Input(shape=(self.config.T, self.config.emb_dim))
        conv2 = Conv1D(self.config.n_filter, kernel_size=2, strides=1, activation="relu")(input)
        conv3 = Conv1D(self.config.n_filter, kernel_size=3, strides=1, activation="relu")(input)
        conv5 = Conv1D(self.config.n_filter, kernel_size=5, strides=1, activation='relu')(input)
        lstm1 = Bidirectional(LSTM(self.config.lstm_dim, dropout=self.config.dropout))(input)
        lstm2 = Bidirectional(LSTM(self.config.lstm_dim, dropout=self.config.dropout))(conv2)
        lstm3 = Bidirectional(LSTM(self.config.lstm_dim, dropout=self.config.dropout))(conv3)
        lstm5 = Bidirectional(LSTM(self.config.lstm_dim, dropout=self.config.dropout))(conv5)

        concat = Concatenate()([lstm1, lstm2, lstm3])
        # bn = BatchNormalization()(concat)
        relu = PReLU()(concat)
        dropout = Dropout(self.config.dropout)(relu)
        dense = Dense(128, kernel_initializer='uniform')(dropout)
        # bn2 = BatchNormalization()(dense)
        # relu2 = PReLU()(dense)
        dropout2 = Dropout(self.config.dropout)(dense)
        out = Dense(1,  activation="sigmoid")(dropout2) #activity_regularizer=l2(self.config.l2_rate), kernel_regularizer=l2(self.config.l2_rate), bias_regularizer=l2(self.config.l2_rate),
        model = Model(inputs=[input], outputs=[out])
        return model

    
    def load_pretrained_model(self, struct_path, weights_path):
        print('loading model .. ')
        with open(struct_path, 'r') as f:
            json = f.read()
        model = keras.models.model_from_json(json)
        model.load_weights(weights_path)
        # self.model.summary()
        return model


class DQN:
    def __init__(self):
        self.

    
if __name__ == '__main__':
    model = Classifier()
    model.load_data()
    model.run_trainer(False)
    