from config import Config
import tushare as ts
import numpy as np
import pandas as pd
import os
import pickle
from collections import defaultdict
class StockEnv:
    def __init__(self, mode):
        self.config = Config()
        self.mode = mode
        self.index_change = defaultdict(list)

    def sz50_code(self):
        codes = ts.get_sz50s()
        return np.array(codes)[:, 1]

    def load_history(self):
        self.history, self.dates, _ = self.prepare_supervision_data()

    def sz50_raw_code(self):
        codes = ts.get_sz50s()
        return np.array(codes)

    def get_initial_state(self, code):
        print("hist length: {}".format(len(self.history[code])))
        return self.history[code][0]

    def get_all_data_of_code(self, code, load=True):
        fnames = os.listdir('./data')
        if '{}.pkl'.format(code) in fnames and load:
            print('loading pickle')
            with open('./data/{}.pkl'.format(code),'rb') as f:
                data_dict = pickle.load(f)
                return data_dict
        print(code)
        data = ts.get_hist_data(code, start=self.config.start_date)
        indices = list(data.columns)
        data_dict = {}
        for index in indices[:]:
            data_dict[index] = np.array(data[index])
        data_dict['date'] = np.array(data.index)
        with open('data/{}.pkl'.format(code),'wb') as f:
            pickle.dump(data_dict, f)
        return data_dict
    
    #TODO
    def retrive_baseline(self, code, t):
        return self.index_change[code][t]
    
    #TODO
    def get_reward(self, profit, baseline):
        return profit - max(0,baseline)
    
    def get_action_profit(self, code, action, t):
        change = self.index_change[code][t]
        quant = action * (change - self.config.cost)
        return quant
        
    #TODO
    def step(self, code, action_ind, t):
        next_state = self.history[code][t+1]
        baseline = self.retrive_baseline(code, t)
        action = self.config.actions[action_ind]
        profit = self.get_action_profit(code, action, t)
        reward = self.get_reward(profit, baseline)
        return next_state, reward

    def temporal_diff(self, d1, d2):
        return (d1[1:] - d2[:-1]) / d2[:-1] * 100.0

    def diff(self, d1, d2):
        return (d1[1:] - d2[1:]) / d2[1:] * 100.0

    def fi_observation(self, data_dict):
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
        print('last date: {}'.format(date[-1]))
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

    def convert_to_training_data(self, code, states):
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
                self.index_change[code].append(change)
                temporal_feature_map.append(states[i:i + self.config.T, :])
        return temporal_feature_map, labels
    
    def prepare_supervision_data(self):
        states, dates, labels = [], [], []
        # print(self.index_change)
        for code in self.config.chosen_stocks:
            data_dict = self.get_all_data_of_code(code)
            substates, subdates = self.fi_observation(data_dict)
            substates, sublabels = self.convert_to_training_data(code, substates)
            # print(np.array(substates)[0:2,:,4])
            # print(substates.shape)
            # print(self.index_change[0:2])
            subdates = subdates[self.config.T:]
            states += substates
            dates += np.ndarray.tolist(subdates)
            labels += sublabels
        states = np.array(states)
        dates = np.array(dates)
        labels = np.array(labels)
        return states, dates, labels
    
    def get_data_today(self, code):
        df = ts.get_realtime_quotes(code)
        return df
    
    def append_state_today(self, data_dict, today_data):
        open = float(today_data['open'][0])
        high = float(today_data['high'][0])
        close = float(today_data['price'][0])
        low = float(today_data['low'][0])
        preclose = float(today_data['pre_close'][0])
        volume = float(today_data['volume'][0])
        change = (float(close) - float(preclose))/float(preclose)*100.0
        ma5 = data_dict['ma5']
        ma10 = data_dict['ma10']
        ma20 = data_dict['ma20']
        vma5 = data_dict['v_ma5']
        vma10 = data_dict['v_ma10']
        vma20 = data_dict['v_ma20']
        _5p = data_dict['close'][-5]
        new_ma5 = (ma5 * 5.0 - _5p + close)/5.0
        _10p = data_dict['close'][-10]
        new_ma10 = (ma10 * 10.0 - _10p + close)/10.0
        _20p = data_dict['close'][-20]
        new_ma20 = (ma20 * 20.0 - _20p + close) / 20.0
        _5v = data_dict['volume'][-5]
        new_vma5 = (vma5 * 5.0 - _5v + volume)/5.0
        _10v = data_dict['volume'][-10]
        new_vma10 = (vma10 * 10.0 - _10v + volume) / 10.0
        _20v = data_dict['volume'][-20]
        new_vma20 = (vma20 * 20.0 - _20v + volume) / 20.0
        # data_dict.append(np.array([open,high,close,low,volume,0,change,new_ma5,new_ma10,new_ma20,new_vma5,new_vma10,new_vma20]), ignore_index=True)
        np.append(data_dict['open'], [open])
        np.append(data_dict['high'], [high])
        np.append(data_dict['close'], [close])
        np.append(data_dict['low'], [low])
        np.append(data_dict['volume'], [volume])
        np.append(data_dict['price_change'], [change])
        np.append(data_dict['p_change'], [change])
        np.append(data_dict['ma5'], [new_ma5])
        np.append(data_dict['ma10'], [new_ma10])
        np.append(data_dict['ma20'], [new_ma20])
        np.append(data_dict['v_ma5'], [new_vma5])
        np.append(data_dict['v_ma10'], [new_vma10])
        np.append(data_dict['v_ma20'], [new_vma20])
        np.append(data_dict['date'],['today'])
        return data_dict


    
    def prepare_prediction_data(self, code):
        data_dict = self.get_all_data_of_code(code)
        current_data = self.get_data_today(code)
        new_data_dict = self.append_state_today(data_dict, current_data)
        # new_data_dict[:-1] = data_dict[1:]
        substates, subdates = self.fi_observation(new_data_dict)
        substates = np.array(substates)[-self.config.T:]
        # print(substates.shape)
        # for i in range(17):
        #     print(substates[:,i])
        # print(substates.shape)
        return substates
    
    