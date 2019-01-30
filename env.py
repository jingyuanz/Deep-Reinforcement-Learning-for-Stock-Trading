from config import Config
import tushare as ts
import numpy as np
import pandas as pd
import os
import pickle
class StockEnv:
    def __init__(self, mode):
        self.config = Config()
        self.mode = mode
        self.index_change = []
        self.history, self.dates, _ = self.prepare_supervision_data()
        print(np.array(self.index_change).shape)

    def sz50_code(self):
        codes = ts.get_sz50s()
        return np.array(codes)[:, 1]

    def sz50_raw_code(self):
        codes = ts.get_sz50s()
        return np.array(codes)

    def get_initial_state(self):
        print("hist length: {}".format(len(self.history)))
        return self.history[0]

    def get_all_data_of_code(self, code):
        fnames = os.listdir('./data')
        if 'data.pkl' in fnames:
            print('loading pickle')
            with open('data/data.pkl','rb') as f:
                data_dict = pickle.load(f)
                return data_dict
        data = ts.get_hist_data(code, start=self.config.start_date)
        indices = list(data.columns)
        data_dict = {}
        for index in indices[:]:
            data_dict[index] = np.array(data[index])
        data_dict['date'] = np.array(data.index)
        with open('data/data.pkl','wb') as f:
            pickle.dump(data_dict, f)
        return data_dict
    
    #TODO
    def retrive_baseline(self, t):
        return self.index_change[t]
    
    #TODO
    def get_reward(self, profit, baseline):
        return profit - max(0,baseline)
    
    def get_action_profit(self, action, t):
        change = self.index_change[t]
        quant = action * change
        return quant
        
    #TODO
    def step(self, action_ind, t):
        next_state = self.history[t+1]
        baseline = self.retrive_baseline(t)
        action = self.config.actions[action_ind]
        profit = self.get_action_profit(action, t)
        reward = self.get_reward(profit, baseline)
        return next_state, reward

    def temporal_diff(self, d1, d2):
        return (d1[1:] - d2[:-1]) / d2[:-1] * 100.0

    def diff(self, d1, d2):
        return (d1[1:] - d2[1:]) / d2[1:] * 100.0

    def fi_observation(self, data_dict):
        keys = data_dict.keys()
        open = data_dict['open']
        high = data_dict['high']
        close = data_dict['close']
        low = data_dict['low']
        volume = data_dict['volume']
        change = data_dict['p_change']
        ma5 = data_dict['ma5']
        ma10 = data_dict['ma10']
        ma20 = data_dict['ma20']
        vma5 = data_dict['v_ma5']
        vma10 = data_dict['v_ma10']
        vma20 = data_dict['v_ma20']
        date = data_dict['date']
        feature_map = np.zeros((len(open) - 1, 17))
        feat_open = self.temporal_diff(open, close)
        feat_high = self.diff(high, open)
        feat_low = self.diff(low, open)
        feat_volume = self.temporal_diff(volume, volume)
        feat_change = change[1:]
        feat_ma5 = ma5[1:] / open[1:]
        feat_delta_ma5 = self.temporal_diff(ma5, ma5)
        feat_ma10 = ma10[1:] / open[1:]
        feat_delta_ma10 = self.temporal_diff(ma10, ma10)
        feat_ma20 = ma20[1:] / open[1:]
        feat_delta_ma20 = self.temporal_diff(ma20, ma20)
        feat_vma5 = vma5[1:] / volume[1:]
        feat_delta_vma5 = self.temporal_diff(vma5, vma5)
        feat_vma10 = vma10[1:] / volume[1:]
        feat_delta_vma10 = self.temporal_diff(vma10, vma10)
        feat_vma20 = vma20[1:] / volume[1:]
        feat_delta_vma20 = self.temporal_diff(vma20, vma20)
        feature_map[:, 0] = feat_open
        feature_map[:, 1] = feat_high
        feature_map[:, 2] = feat_low
        feature_map[:, 3] = feat_volume
        feature_map[:, 4] = feat_change
        feature_map[:, 5] = feat_ma5
        feature_map[:, 6] = feat_delta_ma5
        feature_map[:, 7] = feat_ma10
        feature_map[:, 8] = feat_delta_ma10
        feature_map[:, 9] = feat_ma20
        feature_map[:, 10] = feat_delta_ma20
        feature_map[:, 11] = feat_vma5
        feature_map[:, 12] = feat_delta_vma5
        feature_map[:, 13] = feat_vma10
        feature_map[:, 14] = feat_delta_vma10
        feature_map[:, 15] = feat_vma20
        feature_map[:, 16] = feat_delta_vma20
        return feature_map, date[1:]

    def convert_to_training_data(self, states):
        labels = []
        temporal_feature_map = []
        if self.mode == 'classification':
            # temporal_feature_map = np.zeros((len(states)-self.config.T-1, self.config.T, self.config.feature_size))
            for i in range(len(states) - self.config.T):
                # print(i+self.config.T)
                temporal_feature_map.append(states[i:i + self.config.T, :])
                label = 1 if states[i + self.config.T, 4] > self.config.greediness else 0
                labels.append(label)
        elif self.mode == 'regression':
            pass
        else:
            for i in range(len(states) - self.config.T):
                change = states[i + self.config.T, 4]
                # print(change)
                self.index_change.append(change)
                temporal_feature_map.append(states[i:i + self.config.T, :])
        return temporal_feature_map, labels

    def prepare_supervision_data(self):
        states, dates, labels = [], [], []
        for code in self.config.chosen_stocks:
            data_dict = self.get_all_data_of_code(code)
            substates, subdates = self.fi_observation(data_dict)
            substates, sublabels = self.convert_to_training_data(substates)
            subdates = subdates[self.config.T:]
            states += substates
            dates += np.ndarray.tolist(subdates)
            labels += sublabels
        states = np.array(states)
        dates = np.array(dates)
        labels = np.array(labels)
        print(states.shape, dates.shape, labels.shape)
        print(np.mean(labels))
        return states, dates, labels
    
    
    