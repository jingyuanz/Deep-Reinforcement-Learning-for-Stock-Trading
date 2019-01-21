import numpy as np
from config import Config
from collections import deque
import keras
from keras.layers import *
from keras.regularizers import l2, l1_l2
from keras import Model
from keras.optimizers import *
from keras.losses import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_util import DataUtil
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



class SupervisedAgent:
    def __init__(self, mode='classification'):
        self.config = Config()
        # self.memory_pool = []
        self.agent_mode = mode
        if self.agent_mode == 'classification':
            self.agent_model = self.build_classification_model()
        else:
            self.agent_model = self.build_regression_model()
        self.du = DataUtil(self.agent_mode)
        self.agent_model.summary()
        self.get_data_for_supervision()
        
    def get_data_for_supervision(self):
        self.X, self.dates, self.Y = self.du.prepare_supervision_data()
        
    
    def build_regression_model(self):
        state = Input((self.config.T, self.config.feature_size))
        conv_outs = []
        for fsize in self.config.filter_sizes:
            bn = BatchNormalization()(state)
            conv = Conv1D(self.config.n_filter, fsize, activation='relu')
            conv_out = conv(bn)
            conv_outs.append(conv_out)
        
        lstm_outs = []
        for feature in [state]+conv_outs:
            bn = BatchNormalization()(feature)
            lstm = LSTM(self.config.lstm_dim, recurrent_dropout=self.config.dropout)
            lstm_out = lstm(bn)
            lstm_outs.append(lstm_out)
        
        concat_features = Concatenate(lstm_outs)
        prelu1 = PReLU()(concat_features)
        dense1 = Dense(300)(prelu1)
        drop1 = Dropout(self.config.dropout)(dense1)
        out = Dense(1, activation='linear', kernel_regularizer=l2(self.config.l2_rate), bias_regularizer=l2(self.config.l2_rate), activity_regularizer=l2(self.config.l2_rate))(drop1)
        model = Model(inputs=[state], outputs=[out])
        # model.compile(optimizer=Nadam(lr=self.config.lr), loss=binary_crossentropy, metrics=['accuracy'])
        model.compile(optimizer=Nadam(lr=self.config.lr), loss=mean_squared_logarithmic_error, metrics=['accuracy'])

        return model
    
    def build_classification_model(self):
        state = Input((self.config.T, self.config.feature_size))
        conv_outs = []
        for fsize in self.config.filter_sizes:
            bn = BatchNormalization()(state)
            conv = Conv1D(self.config.n_filter, fsize, activation='relu')
            conv_out = conv(bn)
            conv_outs.append(conv_out)
    
        lstm_outs = []
        for feature in [state] + conv_outs:
            bn = BatchNormalization()(feature)
            lstm = LSTM(self.config.lstm_dim, recurrent_dropout=self.config.dropout)
            lstm_out = lstm(bn)
            lstm_outs.append(lstm_out)
        
        concat_features = Concatenate()(lstm_outs)
        prelu1 = PReLU()(concat_features)
        drop1 = Dropout(self.config.dropout)(prelu1)
        dense1 = Dense(300)(drop1)
        drop2 = Dropout(self.config.dropout)(dense1)
        # out = Dense(1, activation='sigmoid', kernel_regularizer=l2(self.config.l2_rate),
        #             bias_regularizer=l2(self.config.l2_rate), activity_regularizer=l2(self.config.l2_rate))(drop2)
        out = Dense(1, activation='sigmoid')(drop2)
        model = Model(inputs=[state], outputs=[out])
        model.compile(optimizer=Nadam(lr=self.config.lr), loss=binary_crossentropy, metrics=['accuracy'])
        return model
    
    def train(self, mode='regression'):
        json = self.agent_model.to_json()
        with open('./model/'+mode+'.json', 'w') as f:
            f.write(json)
        check = keras.callbacks.ModelCheckpoint('./model/'+mode+'.h5',
                                                monitor='val_acc', verbose=1,
                                                save_best_only=True, save_weights_only=True, mode='auto', period=1)
        self.agent_model.fit([self.X], [self.Y],
                       batch_size=self.config.batch_size,
                       epochs=self.config.epochs,
                       verbose=1,
                       callbacks=[check], validation_split=0.1)
    
    
if __name__ == '__main__':
    agent = SupervisedAgent()
    agent.get_data_for_supervision()
    agent.train()