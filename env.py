from config import Config
import tushare as ts
import numpy as np
import pandas as pd

class StockEnv:
    def __init__(self):
        self.config = Config

    def sz50_code(self):
        codes = ts.get_sz50s()
        return np.array(codes)[:, 1]

    def sz50_raw_code(self):
        codes = ts.get_sz50s()
        return np.array(codes)

    def sample_state(self, code):
        pass

    def get_all_data_of_code(self, code):
        data = ts.get_hist_data(code, start='2018-01-01')
        indices = list(data.columns)
        data_dict = {}
        for index in indices[:]:
            data_dict[index] = np.array(data[index])
        data_dict['date'] = np.array(data.index)
        return data_dict
    
    
    