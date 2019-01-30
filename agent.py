import numpy as np
from config import Config
from collections import deque
import keras
from keras.layers import *
from keras.regularizers import l2, l1_l2
from keras import Model
from keras.optimizers import *
from keras.losses import *
from env import StockEnv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from numpy.random import choice
# from sklearn.utils.random import choice
from collections import OrderedDict
import os
from keras.models import load_model, save_model, model_from_json
from matplotlib.pyplot import plot
from matplotlib import pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def auc(y_true, y_pred):
    # roc = roc_auc_score(y_true, y_pred)
    tf.initialize_local_variables()
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def plot_regression_test(x, Y):
    lenY = len(Y)
    colors = ['red','green','blue','yellow'][:lenY]
    for y,color in list(zip(Y,colors)):
        plot(x, y, color=color)
    plt.show()

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


class QAgent:
    def __init__(self, mode='q'):
        self.config = Config()
        self.fund = self.config.init_fund
        self.memory_pool = []
        self.agent_mode = mode
        self.agent_model = self.build_q_model()
        self.env = StockEnv(mode)
        self.true_history = self.env.prepare_supervision_data()
        self.agent_model.summary()
        self.t = 0
        json = self.agent_model.to_json()
        with open(self.config.q_json, 'w') as f:
            f.write(json)
        self.check = keras.callbacks.ModelCheckpoint(self.config.q_weights,
                                                monitor='loss', verbose=1,
                                                save_best_only=False, save_weights_only=True, mode='auto', period=1)

    def build_q_model(self):
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
        dense1 = Dense(300)(prelu1)
        drop1 = Dropout(self.config.dropout)(dense1)
        out = Dense(self.config.action_size, activation='linear', kernel_regularizer=l2(self.config.l2_rate),
                    bias_regularizer=l2(self.config.l2_rate), activity_regularizer=l2(self.config.l2_rate))(drop1)
        model = Model(inputs=[state], outputs=[out])
        # model.compile(optimizer=Nadam(lr=self.config.lr), loss=binary_crossentropy, metrics=['accuracy'])
        model.compile(optimizer=Nadam(lr=self.config.lr), loss=mean_squared_error)
        return model
    
    def add_to_pool(self, state, action, reward, next_state):
        self.memory_pool.append((state, action, reward, next_state))
        self.t += 1
    
    def epsilon_greedy(self, state):
        self.config.epsilon = self.config.epsilon * self.config.decay
        is_random = choice([0,1], 1, p=[1-self.config.epsilon, self.config.epsilon])
        if is_random:
            random_action = choice(range(self.config.action_size))
            return random_action
        action = self.agent_model.predict(np.array([state]))
        # print(action)
        greedy_action = np.argmax(action)
        # print(greedy_action)
        return greedy_action
    
    def train_by_replay(self):
        indices = range(len(self.memory_pool))
        chosen_indices = choice(indices, size=self.config.batch_size, replace=False)
        memory_batch = np.array(self.memory_pool)[chosen_indices]
        # print(memory_batch[0])
        states, targets = [], []
        tmp_model = keras.models.clone_model(self.agent_model)
        tmp_model.set_weights(self.agent_model.get_weights())
        for mem in memory_batch:
            state = mem[0]
            action = mem[1]
            reward = mem[2]
            next_state = mem[3]
            # print(next_state, next_state.shape)
            value = reward + (self.config.gamma*np.max(tmp_model.predict(np.array([next_state]))[0]))
            target = self.agent_model.predict(np.array([state]))[0]
            target[action] = value
            states.append(state)
            targets.append(target)
        states = np.array(states)
        targets = np.array(targets)
        # print(states.shape, targets.shape)
        self.agent_model.fit(states, targets, batch_size=self.config.batch_size, verbose=0, callbacks=[self.check])
    
    def train(self):
        for i in range(self.config.epochs):
            state = self.env.get_initial_state()
            print("epochs: {}/{}".format(i, self.config.epochs))
            for t in range(len(self.env.history)-1):
                if t%10 == 0:
                    print("\tstep: {}/{}".format(t, len(self.env.history)-1))
                action_ind = self.epsilon_greedy(state)
                next_state, reward = self.env.step(action_ind, t)
                self.add_to_pool(state, action_ind, reward, next_state)
                if len(self.memory_pool)>self.config.MAX_POOL_SIZE:
                    self.memory_pool = self.memory_pool[-self.config.MAX_POOL_SIZE:]
                state = next_state
                if len(self.memory_pool) > self.config.MIN_POOL_SIZE and t%self.config.batch_size==0:
                    self.train_by_replay()
                
    def evaluate(self, agent=True, baseline=True, random=True):
        self.load_trained_agent_model(self.config.q_json, self.config.q_weights)
        FUND = 100000

        baseline_fund = 100000
        random_fund = 100000
        agent_fund = 100000
        
        baseline_trace = []
        random_trace = []
        agent_trace = []
        states = self.env.history[:-1]
        action_probs = self.agent_model.predict(np.array(states), batch_size=128, verbose=1)
        print(action_probs.shape)
        actions = np.argmax(action_probs, axis=-1)
        for t in range(len(self.env.history)-1):
            change = self.env.index_change[t]
            # print(change)
            print("Step : {}/{}, change: {}%".format(t, len(self.env.history)-1, change))
            
            if agent:
                action = self.config.actions[actions[t]]
                buy = action*min(FUND, agent_fund)
                buy_return = (1.0+change/100) * buy
                remain = agent_fund - buy
                agent_fund = remain + buy_return
                agent_trace.append(agent_fund)
                print("\tagent chose action: {},   agent fund: {}".format(action, agent_fund))
            if random:
                random_act = choice(self.config.actions)
                random_buy = random_act * random_fund
                random_buy_return = (1.0+change/100) * random_buy
                random_remain = random_fund - random_buy
                random_fund = random_remain + random_buy_return
                random_trace.append(random_fund)
                print("\tidiot random agent fund: {}".format(random_fund))
            if baseline:
                baseline_fund *= (1.0+change/100)
                baseline_trace.append(baseline_fund)
                print("\tbaseline func: {}".format(baseline_fund))
            print()
        Y = []
        x = range(len(self.env.history)-1)
        if baseline_trace:
            Y.append(baseline_trace)
        if random_trace:
            Y.append(random_trace)
        if agent_trace:
            Y.append(agent_trace)
        
        plot_regression_test(x, Y)
    
    def load_trained_agent_model(self, json_path, weights_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            json = f.read()
        self.agent_model = model_from_json(json)
        self.agent_model.load_weights(weights_path)
        
        
    
class SupervisedAgent:
    def __init__(self, mode='classification'):
        self.config = Config()
        self.agent_mode = mode
        if self.agent_mode == 'classification':
            self.agent_model = self.build_classification_model()
        else:
            self.agent_model = self.build_regression_model()
        self.agent_model.summary()
        self.env = StockEnv(mode)
        self.get_data_for_supervision()
        
    def get_data_for_supervision(self):
        self.X, self.dates, self.Y = self.env.prepare_supervision_data()
        
    
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
        model.compile(optimizer=Nadam(lr=self.config.lr), loss=mean_squared_logarithmic_error)
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