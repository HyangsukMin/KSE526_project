#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lightgbm
from workalendar.asia import SouthKorea # 한국의 공휴일, version : 1.1.1
from utils import WindowGenerator, save_results
#%%
class Config():
    def __init__(
        self,
        input_width = 14,
        label_width = 7,
        shift = 7,
        label_columns = ["Maximum_Power_This_Year"],
        batch_size = 32,
        features = ["meteo", "covid", "gas", "exchange"],#, "gas", "exchange"], #"exchange"], #"gas", ],
        filters = 64,
        kernel_size = 3, 
        activation = 'relu',
        lstm_units = 100,
        attn_units = 100,
        learning_rate = 0.001,
        epochs = 1000,
        verbose = 0,
        aux1 = False,
        aux2 = False,
        is_x_aux1 = False,
        is_x_aux2 = False,
        trial = "linaer_regresson"
        ):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.label_columns = label_columns
        self.batch_size = batch_size
        self.features = features
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.lstm_units = lstm_units
        self.attn_units = attn_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.aux1 = aux1
        self.aux2 = aux2
        self.is_x_aux1 = is_x_aux1
        self.is_x_aux2 = is_x_aux2
        self.trial = trial
config = Config()

data = WindowGenerator(
                            input_width = config.input_width, 
                            label_width = config.label_width, 
                            shift = config.shift, 
                            label_columns = config.label_columns,
                            batch_size = config.batch_size,
                            features = config.features
)
#%%
X_train, y_train = data.train
X_test, y_test  = data.test

X_train = X_train.reshape((-1,X_train.shape[1]*X_train.shape[2]))
y_train = np.squeeze(y_train, axis=-1)

X_test = X_test.reshape((-1,X_test.shape[1]*X_test.shape[2]))
y_test = np.squeeze(y_test, axis=-1)
#%%
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_train, y_train)
#%%
y_pred = data.inverse_transform(lr.predict(X_test))
y_true = data.inverse_transform(y_test)
# %%
import math
def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(((y_true-y_pred)**2).mean())
root_mean_squared_error(y_true[:,-1], y_pred[:,-1]) / (24*7)

#%%
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(y_true[:,-1], y_pred[:,-1]) / (24*7)

# %%
plt.plot(y_pred[:,-1])
plt.plot(y_true[:,-1])
save_results(config, "./outputs/ablation/lr_result.csv")
# %%
import os
save_results(config, y_pred)


# %%
