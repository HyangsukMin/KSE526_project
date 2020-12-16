#%%
from models import CNNBiLSTMATTN, CNNs, LSTMs, CNNLSTM
from models import root_mean_squared_error, weighted_root_mean_squared_error, last_time_step_rmse
from utils import WindowGenerator
from utils import draw_plot, draw_plot_all, save_results
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import math 

import plotly.express as px
import plotly.graph_objects as go

###################################################################
# Config
###################################################################
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
        trial = "CNN_LSTM"
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

callback = tf.keras.callbacks.EarlyStopping(monitor='val_last_time_step_rmse', patience=30)
tf.keras.backend.set_floatx('float64')
###################################################################
# LSTM LSTM
###################################################################
Dataset = WindowGenerator(
                            input_width = config.input_width, 
                            label_width = config.label_width, 
                            shift = config.shift, 
                            label_columns = config.label_columns,
                            batch_size = config.batch_size,
                            features = config.features
)
X_train, y_train = Dataset.train
model_mcge = CNNLSTM(config)
# model_mcge = LSTMs(config)
# model_mcge = CNNLSTM(config)

tf.keras.backend.clear_session()
np.random.seed(1)
tf.random.set_seed(1)
#%%
model_mcge.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    loss=root_mean_squared_error, 
                    metrics=[last_time_step_rmse])
model_mcge.fit(X_train, y_train, epochs=config.epochs, 
                    verbose=config.verbose, 
                    validation_split=0.2,
                    callbacks=[callback]
                    )
#%%
X_test, y_test = Dataset.test
evalutation_m = model_mcge.evaluate(X_test, y_test)
print("Evaluation",evalutation_m)
#%%
y_pred1 = model_mcge.predict(X_train)
y_pred1 = Dataset.inverse_transform(y_pred1)
# y_pred2 = model_mcge.predict(Dataset.val)
# y_pred2 = Dataset.inverse_transform(y_pred2)
y_pred3 = model_mcge.predict(X_test)
y_pred3 = Dataset.inverse_transform(y_pred3)

draw_plot_all(config, Dataset, y_pred1=y_pred1, y_pred3=y_pred3)
#%%
y_true = Dataset.inverse_transform(y_test)
y_pred = Dataset.inverse_transform(model_mcge(X_test))
root_mean_squared_error(Dataset.inverse_transform(
    model_mcge.predict(X_test)),
    Dataset.inverse_transform(Dataset.test[1].reshape((-1,7))))/(y_pred3.shape[0]*7)
#%%
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(Dataset.inverse_transform(
    model_mcge.predict(X_test)),
    Dataset.inverse_transform(Dataset.test[1].reshape((-1,7))))/(y_pred3.shape[0]*7)
#%%
save_results(config, y_pred)
# %%
