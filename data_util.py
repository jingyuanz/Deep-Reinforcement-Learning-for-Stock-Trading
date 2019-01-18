from config import Config
import numpy as np
import tushare as ts
# import cPickle as pickle
from sklearn.utils import shuffle
import pickle
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
from env import StockEnv
class DataUtil:
    def __init__(self, mode):
        self.config = Config()
        self.env = StockEnv()
        self.mode = mode
        
        
    def temporal_diff(self, d1, d2):
        return (d1[1:]-d2[:-1])/d2[:-1]*100.0
    
    def diff(self, d1, d2):
        return (d1[1:]-d2[1:])/d2[1:]*100.0
    
    def fi_observation(self, data_dict):
        keys = data_dict.keys()
        print(keys)
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
        feature_map = np.zeros((len(open)-1, 17))
        feat_open = self.temporal_diff(open, close)
        feat_high = self.diff(high, open)
        feat_low = self.diff(low, open)
        feat_volume = self.temporal_diff(volume, volume)
        feat_change = change[1:]
        feat_ma5 = ma5[1:]/open[1:]
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
        feature_map[:,0] = feat_open
        feature_map[:,1] = feat_high
        feature_map[:,2] = feat_low
        feature_map[:,3] = feat_volume
        feature_map[:,4] = feat_change
        feature_map[:,5] = feat_ma5
        feature_map[:,6] = feat_delta_ma5
        feature_map[:,7] = feat_ma10
        feature_map[:,8] = feat_delta_ma10
        feature_map[:,9] = feat_ma20
        feature_map[:,10] = feat_delta_ma20
        feature_map[:,11] = feat_vma5
        feature_map[:,12] = feat_delta_vma5
        feature_map[:,13] = feat_vma10
        feature_map[:,14] = feat_delta_vma10
        feature_map[:,15] = feat_vma20
        feature_map[:,16] = feat_delta_vma20
        return feature_map, date[1:], []

    def expand_to_2D_states(self, states):
        labels = []
        temporal_feature_map = np.zeros((len(states)-self.config.T-1, self.config.T, self.config.feature_size))
        for i in range(len(states)-self.config.T):
            temporal_feature_map[i,:,:] = states[i:i+self.config.T, :]
            label = 1 if states[i+self.config.T,4]>self.config.greediness else 0
    
    def prepare_supervision_data(self, mode):
        states, dates, labels = [], [], []
        for code in self.config.chosen_stocks:
            data_dict = self.env.get_all_data_of_code(code)
            substates, subdates, sublabels = self.fi_observation(data_dict)
            substates = self.expand_to_2D_states(substates)
            
            states += substates
            dates += subdates
            labels += sublabels
        return states, dates, labels
    
        
if __name__ == '__main__':
    du = DataUtil()
    du.prepare_states()
    # data = du.get_data_from_codes(du.sz50_code(), False)
    # data = ts.get_hist_data('600519', start='2013-01-05')
    # du.get_individual_data('600519')
    # print(data)
