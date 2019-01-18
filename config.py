class Config:
    def __init__(self):
        self.batch_size = 128
        self.data_path = 'data/sz50_day_hist.data'
        self.hist_length = 400
        self.T = 5
        self.emb_dim = 13
        self.lstm_dim = 150
        self.n_filter = 150
        self.filter_size = 2
        self.dropout = 0.5
        self.lr = 3e-4
        self.epochs = 15
        self.gamma = 0.1
        self.result_path = 'data/results.txt'
        self.final_round_model_path = 'model/final.model'
        self.val_best_model_path = 'model/val.model'
        self.predict_model_path = 'model/2018-09-03_best.model'
        self.train_data_path = 'data/train.data'
        self.test_data_path = 'data/test.data'
        self.greediness = 0.008
        self.start_date = '2015-01-01'
        self.chosen_stocks = ['sh']