from models import CNNBiLSTMATTN_noAuxs, CNNBiLSTMATTN_noAUX1, CNNBiLSTMATTN_noAUX2, CNNBiLSTMATTN, BiLSTMATTN
from models import root_mean_squared_error, weighted_root_mean_squared_error, last_time_step_rmse
from utils import WindowGenerator
from utils import draw_plot, draw_plot_all
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

#%%
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
        features = ["meteo", "covid", "gas", "exchange"],
        filters = 64,
        kernel_size = 3, 
        activation = 'relu',
        lstm_units = 100,
        attn_units = 100,
        learning_rate = 0.001,
        epochs = 200,
        verbose = 1,
        aux1 = True,
        aux2 = True
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
config = Config()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_last_time_step_rmse', patience=100)
tf.keras.backend.set_floatx('float64')
###################################################################
# Model
###################################################################
# model_mcge = CNNBiLSTMATTN_noAUX2(config)
# model_mcge = CNNBiLSTMATTN_noAUX1(config)
model_mcge = CNNBiLSTMATTN_noAuxs(config)

#%%
###################################################################
# CNN BiLSTM ATTN
###################################################################
Dataset = WindowGenerator(
                            input_width = config.input_width, 
                            label_width = config.label_width, 
                            shift = config.shift, 
                            label_columns = config.label_columns,
                            batch_size = config.batch_size,
                            features = config.features,
                            aux1 = config.aux1,
                            aux2 = config.aux2
)
#%%

tf.keras.backend.clear_session()
np.random.seed(1)
tf.random.set_seed(1)

model_mcge.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    loss=[weighted_root_mean_squared_error,'mape'],# loss 뭐쓰징 ~
                    metrics=[last_time_step_rmse,'mape'])
# X_train, y_train = Dataset.train
# X_val, y_val = Dataset.val
X_train, X_train_aux1, X_train_aux2, y_train = Dataset.train
model_mcge.fit(
                [X_train, X_train_aux1, X_train_aux2], 
                y_train, 
                epochs=config.epochs, 
                verbose=config.verbose, 
                validation_split = 0.2,
                batch_size = config.batch_size,
                steps_per_epoch = 1,
                callbacks=[callback],
                )
#%%
X_test, X_test_aux1, X_test_aux2, y_test = Dataset.test
evalutation_m = model_mcge.evaluate([X_test, X_test_aux1, X_test_aux2], y_test)
print("Evaluation",evalutation_m)
#%%
y_pred = model_mcge.predict([X_test, X_test_aux1, X_test_aux2])
y_pred = Dataset.inverse_transform(y_pred)
y_pred1 = model_mcge.predict([X_train, X_train_aux1, X_train_aux2])
y_pred1 = Dataset.inverse_transform(y_pred1)
# y_pred2 = model_mcge.predict([X_val, X_val_aux])
# y_pred2 = Dataset.inverse_transform(y_pred2)
y_pred3 = model_mcge.predict([X_test, X_test_aux1, X_test_aux2])
y_pred3 = Dataset.inverse_transform(y_pred3)

draw_plot_all(config,Dataset, y_pred1 = y_pred1, y_pred3 = y_pred3)
#%%
root_mean_squared_error(Dataset.inverse_transform(model_mcge.predict([X_test, X_test_aux1, X_test_aux2])),
                        Dataset.inverse_transform(Dataset.test[3].reshape((-1,7))))/(y_pred3.shape[0]*7)
#%%
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(Dataset.inverse_transform(model_mcge.predict([X_test, X_test_aux1, X_test_aux2])),
                        Dataset.inverse_transform(Dataset.test[3].reshape((-1,7))))/(y_pred3.shape[0]*7)
#%%
