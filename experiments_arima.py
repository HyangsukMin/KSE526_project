#%%
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller #ADF검정 -> 정상성 확인
import matplotlib.pyplot as plt
from utils import save_results
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
        trial = "ARIMA"
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

#%%
def creat_dataset(data, n_in = 21, n_out = 7):
    # [samples, timesteps, features]
    X = np.empty(shape=(0,n_in,data.shape[-1]))
    Y = np.empty(shape=(0,n_out,2))
    for i in range(len(data) - (n_in + n_out)):
        x = data[i:(i+n_in)].values
        y = data.loc[(i+n_in):(i+n_in+n_out-1),["Date","Maximum_Power_This_Year"]].values
        X = np.vstack([X,np.array(x)[np.newaxis,...]])
        Y = np.vstack([Y,np.array(y)[np.newaxis,...]])
    # X = np.array(X)
    # Y = np.array(Y)
    return X, Y

def create_dataset_holiday(data, n_in = 21, n_out = 7):
    X = np.empty(shape=(0,n_out,2))
    for i in range(len(data) - (n_in + n_out)):
        x = data.loc[(i+n_in):(i+n_in+n_out-1),["weekday","holiday"]].values
        X = np.vstack([X, np.array(x)[np.newaxis,...]])
    return X

def split_data(X, Y = None, test_size = 14):
    if Y is None :
        X_train, X_test = X[:(len(X)-test_size)], X[(len(X)-test_size):]
        print("X_train shape : {}\n X_test shape : {}".format(X_train.shape, X_test.shape))
        return X_train, X_test
    else :
        X_train, X_test = X[:(len(X)-test_size)], X[(len(X)-test_size):]
        y_train, y_test = Y[:(len(Y)-test_size)], Y[(len(Y)-test_size):]
        print("X_train shape : {}, y_train shape : {}\nX_test shape : {}, y_test shape : {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
        # train = np.array(np.split(train, len(train)/7))
        # test = np.array(np.split(test, len(test)/7))
        return X_train, y_train, X_test, y_test
#%%
kpx = pd.read_csv("./data/preprocess/KPX_load.csv")
data = kpx.iloc[-329:,5].reset_index(drop=True)
mean = data.mean()
std = data.std()
data = (data - mean)/std
#%%
series = data.iloc[:-24]
plot_acf(series)
plot_pacf(series)
plt.show()
result = adfuller(series)
print(f'원 데이터 ADF Statistic: {result[0]:.3f}')
print(f'원 데이터 p-value: {result[1]:.3f}')
#%%
diff_1=series.diff(periods=1).iloc[1:]
diff_1.plot()
plot_acf(diff_1)
plot_pacf(diff_1)
plt.show()
result = adfuller(diff_1)
print(f'1차 차분 ADF Statistic: {result[0]:.3f}')
print(f'1차 차분 p-value: {result[1]:.3f}')
# %%
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(series, order=(2,1,1))
model_fit = model.fit(trend='nc',full_output=True, disp=1)
print(model_fit.summary())
# %%
model_fit.plot_predict()
# %%
y_pred = (model_fit.forecast(steps=24)[0] * std) + mean
y_true = (data.iloc[-24:].reset_index(drop=True) * std) + mean
# %%
plt.plot(y_pred)
plt.plot(y_true)
plt.show()
# %%
import math
def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(((y_true-y_pred)**2).mean())
root_mean_squared_error(y_true, y_pred) / 24
#%%
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(y_true, y_pred) / 24
# %%
# %%
import os
def save_results(config,y_pred):
    sub = pd.DataFrame(data=y_pred, index=pd.date_range(start="2020.11.01",end="2020.11.24",freq="d"))
    output_dir = "./outputs"
    fname = "final_model_output_"+str(config.trial)+"_"+str("-".join(config.features))+"_"+str(config.is_x_aux1)+"_"+str(config.is_x_aux2)+".csv"
    sub.to_csv(os.path.join(output_dir, fname))
save_results(config, y_pred)
# %%
