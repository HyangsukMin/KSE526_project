#%%
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
pd.set_option('display.max_columns',1500)
#%%
df_Load = pd.read_csv("./data/preprocess/KFX_Load.csv")
df_Load[df_Load.Maximum_Power_This_Year == "5.307.4"] = "5307.4"

px.line(df_Load,x="Date",y="Maximum_Power_This_Year")
# %%
df_SMP = pd.read_csv("./data/preprocess/KFX_SMP_hourly.csv")
df_SMP["Date"] = pd.to_datetime(df_SMP["Date"],format="%Y%m%d")
px.line(df_SMP.sort_values(["Date","Times"])[df_SMP.Times == "12h"],x="Date",y="SMP",width=1000)
# %%



#%%
#####################################################
# Linear Regression
#####################################################
def dataset(df, w = 14):
    tmp = df["Maximum_Power_This_Year"].str.replace(',','').apply(float)
    tmp[tmp == "5.307.4"] = "5307.4"
    X_train = []
    y_train = []
    for i in range(0,len(tmp)-15):
        x = tmp[i:(i+w)].values.astype(float)
        y = float(tmp[(i+w+1)])
        X_train.append(x.tolist())
        y_train.append(y)
    return X_train, y_train

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
X_train, y_train = dataset(df_Load,w=14)
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train)
lr = LinearRegression()
lr.fit(X_train, y_train)
# lr.predict(X_test).shape
mean_squared_error(y_test, lr.predict(X_test))
plt.plot(y_test)
plt.plot(lr.predict(X_test))

# %%
##################################################
# CNN-LSTM Encoder-Decoder With Univariate Input
###################################################
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import matplotlib.pyplot as plt

def dataset(df, w = 8):
    tmp = df["Maximum_Power_This_Year"].str.replace(',','').apply(float)
    tmp[tmp == "5.307.4"] = "5307.4"
    X = []
    for i in range(0,len(tmp)-15):
        x = tmp[i:(i+w)].values.astype(float)
        X.append(x.tolist())
    return np.array(X)

def split_dataset(data):
    train, test = data[1:-602], data[-602:]
    print(train.shape, test.shape)
    train = np.array(np.split(train, len(train)/7))
    test = np.array(np.split(test, len(test)/7))
    return train, test

def evaluate_forecasts(actual, predicted):
    scores = list()
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:,i], predicted[:,i])
        rmse = math.sqrt(mse)
        scores.append(rmse)
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%3.f] %s' % (name, score, s_scores))

def to_supervised(train, n_input, n_out = 7):
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0

    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out

        if out_end < len(data):
            x_input = data[in_start:in_end,0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return np.array(X), np.array(y)

def build_model(train, n_input):
    train_x, train_y = to_supervised(train, n_input)
    verbose, epochs, batch_size = 1, 50, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1],1))
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.RepeatVector(n_outputs))
    model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(100, activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_x, train_y, epochs=epochs, batch_size = batch_size, verbose= verbose)
    return model

def forecast(model, history, n_input):
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    input_x = data[-n_input:, 0]
    input_x = input_x.reshape((1, len(input_x), 1))
    yhat = model.predict(input_x, verbose=0)
    yhat = yhat[0]
    return yhat

def evalueate_model(train, test, n_input):
    model = build_model(train, n_input)
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i,:])
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


X = dataset(df_Load)
train, test = split_dataset(X)
n_input = 7
score, scores = evalueate_model(train, test, n_input)
summarize_scores('cnn', score, scores)
days = ['sun','mon','tue','wed','thr','fri','sat']
plt.plot(days, scores, marker='o', label = 'lstm')
plt.plot()

# summarize_scores('cnn', score, scores)
# days = ['sun','mon','tue','wed','thr','fri','sat']
# plt.plot(days, scores, marker='o', label = 'lstm')
# plt.plot()
# %%
