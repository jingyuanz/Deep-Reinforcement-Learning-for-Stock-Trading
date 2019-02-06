from matplotlib.pyplot import plot
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

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
