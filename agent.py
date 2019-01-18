import numpy as np
from config import Config
from collections import deque
import keras
from keras.layers import *
from keras.regularizers import l2, l1_l2
from keras import Model
from keras.optimizers import *
from keras.losses import *
from data_util import DataUtil
class SupervisedAgent:
    def __init__(self):
        self.config = Config()
        self.du = DataUtil()
        # self.memory_pool = []
        self.agent_mode = 'classification'
        if self.agent_mode == 'classification':
            self.agent_model = self.build_classification_model()
        else:
            self.agent_model = self.build_regression_model()
        self.agent_model.summary()
        
    def get_data_for_supervision(self):
        self.X, self.dates, self.Y = self.du.prepare_states(self.agent_mode)
    
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
        
        concat_features = Concatenate(lstm_outs)
        prelu1 = PReLU()(concat_features)
        dense1 = Dense(300)(prelu1)
        drop1 = Dropout(self.config.dropout)(dense1)
        out = Dense(1, activation='sigmoid', kernel_regularizer=l2(self.config.l2_rate),
                    bias_regularizer=l2(self.config.l2_rate), activity_regularizer=l2(self.config.l2_rate))(drop1)
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