#%%
from models import CNNBiLSTMATTN, BiLSTMATTN, LSTMaux, CNNLSTM, CNNLSTMATTN, LSTMATTN
from models import root_mean_squared_error, weighted_root_mean_squared_error, last_time_step_rmse
from utils import WindowGenerator
from utils import draw_plot, draw_plot_all, save_results
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
        features = ["meteo", "covid", "gas", "exchange"],#, "gas", "exchange"], #"exchange"], #"gas", ],
        filters = 64,
        kernel_size = 3, 
        activation = 'relu',
        lstm_units = 100,
        attn_units = 100,
        learning_rate = 0.001,
        epochs = 1000,
        verbose = 0,
        aux1 = True,
        aux2 = True,
        is_x_aux1 = True,
        is_x_aux2 = True,
        trial = "cnn_fcn_model_revise"
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

callback = tf.keras.callbacks.EarlyStopping(monitor='val_last_time_step_rmse', patience=100)
tf.keras.backend.set_floatx('float64')
#%%
# model_mcge = BiLSTMATTN(config)
model_mcge = CNNBiLSTMATTN(config)
#%%
###################################################################
# KPX x Meteorology x COVID-19 x gasoline x exchange
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
                    loss=[root_mean_squared_error,'mape'],# loss 뭐쓰징 ~
                    metrics=[last_time_step_rmse,'mape'])

X_train, X_train_aux1, X_train_aux2, y_train = Dataset.train
print(X_train.shape)
print(X_train_aux1.shape)
print(X_train_aux2.shape)
history = model_mcge.fit(
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

import pandas as pd
# pd.DataFrame(data=y_pred3,columns=pd.date_range('2020.11.01', end='2020.11.24', freq='d')).to_csv("./outputs/mainModel.csv",index=False)

draw_plot_all(config, Dataset, y_pred1 = y_pred1, y_pred3 = y_pred3)
#%%
root_mean_squared_error(Dataset.inverse_transform(model_mcge.predict([X_test, X_test_aux1, X_test_aux2])),
                        Dataset.inverse_transform(Dataset.test[3].reshape((-1,7))))/(y_pred3.shape[0]*7)
#%%
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(Dataset.inverse_transform(model_mcge.predict([X_test, X_test_aux1, X_test_aux2])),
                                Dataset.inverse_transform(Dataset.test[3].reshape((-1,7))))/(y_pred3.shape[0]*7)
#%%
print(y_pred3.shape)
print(Dataset.inverse_transform(Dataset.test[3]).reshape((-1,7)).shape)
#%%
# save_results(config,y_pred3)
#%%
# ###################################################################
# # KPX
# ###################################################################
# kpx_data = WindowGenerator(input_width = input_width, 
#                             label_width = label_width, 
#                             shift = label_width, 
#                             label_columns = label_columns,
#                             batch_size = batch_size)
# #%%
# model = CNNLSTMATTN(n_outputs = label_width, 
#                 # conv = 2, 
#                 # lstm = 3, 
#                 filters = 64, 
#                 kernel_size = 3, 
#                 activation = 'relu', 
#                 lstm_units = 30)
# #%%
# tf.keras.backend.clear_session()
# np.random.seed(1)
# tf.random.set_seed(1)

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error, metrics=[last_time_step_rmse])
# model.fit(kpx_data.train, epochs=epochs, 
#                     verbose=verbose, 
#                     validation_data = kpx_data.val,
#                     callbacks=[callback]
# )

# #%%
# evalutation = model.evaluate(kpx_data.test)
# print("Evaluation",evalutation)
# kpx_data.labels.values
# y_pred = model.predict(kpx_data.test)
# y_pred = kpx_data.inverse_transform(y_pred)

# #%%
# y_pred1 = model.predict(kpx_data.train)
# y_pred1 = kpx_data.inverse_transform(y_pred1)
# y_pred2 = model.predict(kpx_data.val)
# y_pred2 = kpx_data.inverse_transform(y_pred2)
# y_pred3 = model.predict(kpx_data.test)
# y_pred3 = kpx_data.inverse_transform(y_pred3)

# draw_plot_all(kpx_data, y_pred1, y_pred2, y_pred3)
# # draw_plot(kpx_data, y_pred3)
# # fig.add_trace(go.Scatter( y = y_pred[:,-3], mode='lines'))



# #%%
# ###################################################################
# # KPX x Meteorology
# ###################################################################
# km_data = WindowGenerator(input_width = input_width, 
#                             label_width = label_width, 
#                             shift = label_width, 
#                             label_columns = label_columns,
#                             batch_size = batch_size,
#                             features = ["meteo"])
# #%%
# model_m = CNNLSTMATTN(n_outputs = label_width, 
#                 # conv = 2, 
#                 # lstm = 3, 
#                 filters = 64, 
#                 kernel_size = 3, 
#                 activation = 'relu', 
#                 lstm_units = 30)
# #%%
# tf.keras.backend.clear_session()
# np.random.seed(1)
# tf.random.set_seed(1)

# model_m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error, metrics=[last_time_step_rmse])
# model_m.fit(km_data.train, epochs=epochs, 
#                     verbose=verbose, 
#                     validation_data = km_data.val,
#                     callbacks=[callback]
#                     )

# #%%
# evalutation_m = model_m.evaluate(km_data.test)
# print("Evaluation",evalutation_m)
# y_pred = model_m.predict(km_data.test)
# y_pred = km_data.inverse_transform(y_pred)

# #%%
# y_pred1 = model_m.predict(km_data.train)
# y_pred1 = km_data.inverse_transform(y_pred1)
# y_pred2 = model_m.predict(km_data.val)
# y_pred2 = km_data.inverse_transform(y_pred2)
# y_pred3 = model_m.predict(km_data.test)
# y_pred3 = km_data.inverse_transform(y_pred3)

# draw_plot_all(km_data, y_pred1, y_pred2, y_pred3)
# draw_plot(km_data, y_pred3)

# #%%
# ###################################################################
# # KPX x Meteorology x COVID-19
# ###################################################################
# kmc_data = WindowGenerator(input_width = input_width, 
#                             label_width = label_width, 
#                             shift = label_width, 
#                             label_columns = label_columns,
#                             batch_size = batch_size,
#                             features = ["meteo", "covid"]
#                             )
# #%%
# model_mc = CNNLSTMATTN(n_outputs = label_width, 
#                 # conv = 2, 
#                 # lstm = 3, 
#                 filters = 64, 
#                 kernel_size = 3, 
#                 activation = 'relu', 
#                 lstm_units = 30)
# #%%
# tf.keras.backend.clear_session()
# np.random.seed(1)
# tf.random.set_seed(1)

# model_mc.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error, metrics=[last_time_step_rmse])
# model_mc.fit(kmc_data.train, epochs=epochs, 
#                     verbose=verbose, 
#                     validation_data = kmc_data.val,
#                     callbacks=[callback]
#                     )

# #%%
# evalutation_m = model_mc.evaluate(kmc_data.test)
# print("Evaluation",evalutation_m)
# y_pred = model_mc.predict(kmc_data.test)
# y_pred = kmc_data.inverse_transform(y_pred)

# #%%
# y_pred1 = model_mc.predict(kmc_data.train)
# y_pred1 = kmc_data.inverse_transform(y_pred1)
# y_pred2 = model_mc.predict(kmc_data.val)
# y_pred2 = kmc_data.inverse_transform(y_pred2)
# y_pred3 = model_mc.predict(kmc_data.test)
# y_pred3 = kmc_data.inverse_transform(y_pred3)

# draw_plot_all(kmc_data, y_pred1, y_pred2, y_pred3)
# draw_plot(kmc_data, y_pred3)
# %%
