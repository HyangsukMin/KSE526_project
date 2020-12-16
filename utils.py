#%%
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from workalendar.asia import SouthKorea # 한국의 공휴일, version : 1.1.1
import plotly.express as px
import plotly.graph_objects as go
import os
# tf.debugging.set_log_device_placement(True)
# filelist = [
#     "./data/COVID/covid19.csv",
#     "./data/preprocess/KFX_load.csv",
#     "./data/preprocess/Meteorology.csv"
# ]
#%%

def save_results(config,y_pred):
    sub = pd.DataFrame(data=y_pred[:,-1], index=pd.date_range(start="2020.11.01",end="2020.11.24",freq="d"))
    output_dir = "./outputs"
    fname = "final_model_output_"+str(config.trial)+"_"+str("-".join(config.features))+"_"+str(config.is_x_aux1)+"_"+str(config.is_x_aux2)+".csv"
    sub.to_csv(os.path.join(output_dir, fname))


def draw_plot(data, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter( x = data.data.Date, y = data.data.Maximum_Power_This_Year, name = "True",
                            line = dict(color="royalblue", width = 2, dash='dashdot')
    ))
    fig.add_trace(go.Scatter( x = data.date_test_df, y = y_pred[:,-1], name = "Pred",
                            line = dict(color="firebrick", width = 4)
    ))
    fig.update_layout(title = "Load Forecasting", 
                        yaxis_title = "Power (10000kW)",
                        xaxis_title = "Date",
                        width = 700)
    fig.show()

def draw_plot_all(config, data, y_pred1 = None, y_pred2 = None, y_pred3 = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter( x = data.data.Date, y = data.data.Maximum_Power_This_Year, name = "True",
                            line = dict(color="royalblue", width = 1.5, dash='dashdot')
    ))
    if y_pred1 is not None:
        fig.add_trace(go.Scatter( x = data.date_train_df[(data.total_window_size):], y = y_pred1[:,-1], name = "Pred1",
                                line = dict(color="yellow", width = 2)
        ))
    if y_pred2 is not None:
        fig.add_trace(go.Scatter( x = data.date_val_df, y = y_pred2[:,-1], name = "Pred2",
                                line = dict(color="green", width = 2)
        ))
    if y_pred3 is not None:
        fig.add_trace(go.Scatter( x = data.date_test_df, y = y_pred3[:,-1], name = "Pred3",
                                line = dict(color="firebrick", width = 2)
        ))
    # fig.add_trace(go.Scatter( x = data.data.Date, y = data.data.Maximum_Power_This_Year, name = "True",
    #                         line = dict(color="royalblue", width = 1.5, dash='dashdot')
    # ))
    if config.features is not None:
        title = "Load Forecasting_"+"_".join(config.features)
    else :
        title = "Load Forecasting_"
    fig.update_layout(title = title, 
                        yaxis_title = "Power (10000kW)",
                        xaxis_title = "Date",
                        width = 700)
    fig.show()

class WindowGenerator():
    def __init__(self, 
                        input_width, 
                        label_width, 
                        shift, 
                        batch_size,
                        label_columns = ["Maximum_Power_This_Year"], 
                        features = None,
                        aux1 = False,
                        aux2 = False):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.label_columns = label_columns
        self.total_window_size = self.input_width + self.shift
        self.aux1 = aux1
        self.aux2 = aux2

        ##############################################################
        # Raw data
        ##############################################################
        kpx_load = pd.read_csv("./data/preprocess/KPX_load.csv")
        self.kpx_load = kpx_load[["Date",
                            # "Installed_Capacity",
                            "Supply_Capacity",
                            "Maximum_Power_Last_Year",
                            "Maximum_Power_This_Year",
                            "Supply_Reserve"
                            ]][-329:].reset_index(drop=True)
        
        self.kpx_load_size = self.kpx_load.shape[1] - 1
        self.data = self.kpx_load
        self.internal_size = self.kpx_load_size
        self.external_size = 0
        if isinstance(features,list):
            if "meteo" in features:
                """
                """
                meteorology = pd.read_csv("./data/preprocess/Meteorology.csv")
                meteorology = meteorology[["location","Date",
                                        "avg_temp","min_temp","max_temp", # 평균기온
                                        "max_rain_1h", #강우량
                                        "avg_dew_point", #이슬점
                                        "avg_relative_humidity", #상대습도
                                        "sunshine_hr", #일조시간
                                        "avg_land_temp", #지면온도
                                        ]]
                meteorology = meteorology.fillna(0).groupby("Date").agg('mean') #'mean'
                self.meteorology = meteorology.drop(["location"], axis=1)
                self.meteorology_size = self.meteorology.shape[1]

                self.internal_size += self.meteorology_size
                self.data = pd.merge(self.data, self.meteorology, on="Date")
                # print(self.data.head(5))
            if "covid" in features:
                """
                """
                covid = pd.read_csv("./data/preprocess/covid19.csv")
                covid["Date"] = covid["Date"].str.replace(" ","")
                covid["Sum_diff"] = np.gradient(covid.Sum,1)
                covid["Sum_diff2"] = np.gradient(covid.Sum,2)
                self.covid = covid[["Date",
                                "Sum_diff2", #전일대비 증가량의 증가량
                                "Sum_diff", #전일대비 증가량
                                # "Sum"
                ]]
                self.covid_size = self.covid.shape[1] - 1 # except Date
                self.external_size += self.covid_size
                self.data = pd.merge(self.data, self.covid, on="Date")
                # print(self.data.head(5))

            if "gas" in features:
                """
                """
                gas = pd.read_csv("./data/preprocess/shell_price.csv")
                gas["gasoline_diff"] = np.gradient(gas.gasoline2,1)
                gas["diesel_diff"] = np.gradient(gas.diesel,1)
                self.gas = gas[["Date",
                            "gasoline2", #일반휘발유 가격
                            "diesel", #경유 가격
                            "gasoline_diff", #일반휘발유 가격 전일대비 증가량
                            "diesel_diff" #경유 전일대비 증가량
                ]]
                self.gas_size = self.gas.shape[1] - 1
                self.external_size += self.gas_size
                self.data = pd.merge(self.data, self.gas, on="Date")
                # print(self.data.head(5))

            if "exchange" in features:
                """
                """
                exchange = pd.read_csv("./data/preprocess/exchange.csv")
                exchange["Last_diff"] = np.gradient(exchange.Last,1)
                self.exchange = exchange[["Date",
                                    "Last", #종가
                                    "Last_diff" #종가 전일 대비 증가량
                ]] 
                self.exchange_size = self.exchange.shape[1] - 1
                self.external_size += self.exchange_size
                self.data = pd.merge(self.data, self.exchange, on="Date")
                # print(self.data.head(5))

        """
        Others : holiday, weekday information
        """
        date = pd.date_range('2020.01.01', end='2020.11.24', freq='d')
        date = pd.DataFrame(columns=["Date"],data=date.astype(str).values)
        date["Date"] = date["Date"].str.replace("-",".")
        date["weekday"] = pd.to_datetime(date["Date"]).dt.weekday
        # week_dict = {0:0,1:1,2:1,3:1,4:1,5:2,6:2}
        # date["weekday"] = date["weekday"].map(week_dict)
        date["holiday"] = 0
        date.loc[date.Date.isin(pd.Series(np.array(SouthKorea().holidays(2020))[:, 0]).map(str).str.replace("-",".")),"holiday"] = 1
        date.loc[date.Date.isin(["2020.01.24", "2020.01.25", "2020.01.26", "2020.01.27"]),"holiday"] = 2
        date.loc[date.Date.isin(["2020.09.30", "2020.10.01", "2020.10.02", "2020.10.03"]),"holiday"] = 2

        ##############################################################
        # dummy
        ##############################################################
        weekday_dum = pd.get_dummies(date.weekday, prefix = "week")
        date = pd.concat([date,weekday_dum],axis=1)
        holiday_dum = pd.get_dummies(date.holiday, prefix = "holiday")
        date = pd.concat([date,holiday_dum],axis=1)
        date.drop(["weekday","holiday"], axis = 1, inplace = True)

        self.date = date
        self.date_size = date.shape[1] - 1

        ################################################################
        # Post-process
        ################################################################
        self.data = pd.merge(self.data, self.date, on="Date", how='left')
        self.data.fillna(0,inplace=True)
        idx_val  = self.data[self.data.Date == "2020.11.01"].index.values[0]
        idx_test = self.data[self.data.Date == "2020.11.01"].index.values[0]

        # divice it into train, val, and test
        # Will create train and test only
        self.train_df = self.data[:idx_val]
        self.val_df = self.data[(idx_val - self.total_window_size):idx_test]
        self.test_df = self.data[(idx_val - self.total_window_size):]

        # get date info
        self.date_train_df = self.train_df.Date
        self.date_val_df = self.val_df.Date[self.total_window_size:]
        self.date_test_df = self.test_df.Date[self.total_window_size:]
        self.train_df.drop(["Date"],axis=1,inplace=True)
        self.val_df.drop(["Date"],axis=1,inplace=True)
        self.test_df.drop(["Date"],axis=1,inplace=True)

        ##############################################################
        # Scaler
        ##############################################################
        # self.data_mean = self.data.drop(["Date"],axis=1).mean()
        # self.data_std = self.data.drop(["Date"],axis=1).std()

        # self.train_df = (self.train_df - self.data_mean) / self.data_std # Normalize
        # self.val_df = (self.val_df - self.data_mean) / self.data_std
        # self.test_df = (self.test_df - self.data_mean) / self.data_std

        self.data_min = self.data.drop(["Date"],axis=1).min()
        self.data_max = self.data.drop(["Date"],axis=1).max()

        self.train_df = (self.train_df - self.data_min) / (self.data_max - self.data_min) # Normalize
        self.val_df = (self.val_df - self.data_min) / (self.data_max - self.data_min)
        self.test_df = (self.test_df - self.data_min) / (self.data_max - self.data_min)

        ##############################################################
        # Work out the label column indices.
        ##############################################################
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(self.train_df.columns)}

        # ##############################################################
        # # Work out the window parameters
        # ##############################################################
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
            
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}'])
    # def split_window(self, features):

    #     inputs = features[:, self.input_slice, :-5]
    #     labels = features[:, self.labels_slice, :-5]
    #     aux = features[:,self.labels_slice,-5:]
    #     if self.label_columns is not None:
    #         labels = tf.stack(
    #             [labels[:, :, self.column_indices[name]] for name in self.label_columns],
    #             axis=-1)
        
    #     # Slicing doesn't preserve static shape information, so set the shapes
    #     # manually. This way the `tf.data.Datasets` are easier to inspect.
    #     inputs.set_shape([None, self.input_width, None])
    #     labels.set_shape([None, self.label_width, None])
    #     aux.set_shape([None, self.label_width, None])
    #     # inputs = {input:inputs, aux:aux}
    #     return inputs, labels

    def create_dataset(self, data, aux1 = False, aux2 = False):
        # [samples, timesteps, features]
        if aux1 == False:
            inputs = np.empty(shape=(0,self.input_width,data.shape[-1]))
            labels = np.empty(shape=(0,self.label_width,1))
            label_columns = self.column_indices[self.label_columns[0]]

            for i in range(len(data) - self.total_window_size):
                data_window = data[i:(i+self.total_window_size)]
                x = data_window.iloc[:self.input_width,:].values
                y = data_window.iloc[-self.label_width:,label_columns].values
                inputs = np.vstack([inputs,np.array(x)[np.newaxis,...]])
                labels = np.vstack([labels,np.array(y)[np.newaxis,...,np.newaxis]])

        else :
            inputs = np.empty(shape=(0,self.input_width, self.internal_size + self.date_size))
            inputs_aux1 = np.empty(shape=(0,self.input_width, self.external_size))
            labels = np.empty(shape=(0,self.label_width,1))
            label_columns = self.column_indices[self.label_columns[0]]

            for i in range(len(data) - self.total_window_size):
                data_window = data[i:(i+self.total_window_size)]
                x_pre = data_window.iloc[:self.input_width,:self.internal_size]
                x_pro = data_window.iloc[:self.input_width,-self.date_size:]
                x = pd.concat([x_pre,x_pro],axis=1)
                x = x.values
                aux = data_window.iloc[:self.input_width,self.internal_size:(self.internal_size+self.external_size)].values
                y = data_window.iloc[-self.label_width:,label_columns].values
                inputs = np.vstack([inputs,np.array(x)[np.newaxis,...]])
                inputs_aux1 = np.vstack([inputs_aux1,np.array(aux)[np.newaxis,...]])
                labels = np.vstack([labels,np.array(y)[np.newaxis,...,np.newaxis]])
    
        if aux2 :
            inputs_aux2 = np.empty(shape=(0, self.label_width, self.date_size))
            for i in range(len(data) - self.total_window_size):
                data_window = data[i:(i+self.total_window_size)]
                aux = data_window.iloc[-self.label_width:,-self.date_size:].values
                inputs_aux2 = np.vstack([inputs_aux2, np.array(aux)[np.newaxis,...]])
            return inputs, inputs_aux1, inputs_aux2, labels
    
        else :
            print(inputs.shape, labels.shape)
            return inputs, labels

    # def make_dataset(self, data, batch_size):
    #     data = np.array(data, dtype=np.float64)
    #     data = tf.convert_to_tensor(data, dtype=tf.float64)
    #     ds = tf.keras.preprocessing.timeseries_dataset_from_array(
    #         data=data,
    #         targets=None,
    #         sequence_length=self.total_window_size,
    #         sequence_stride=1,
    #         shuffle=False,
    #         batch_size=batch_size
    #         )
    #     ds = ds.map(self.split_window)
    #     return ds

    def inverse_transform(self, y):
        return ( y * (self.data_max[self.label_columns].values[0] - self.data_min[self.label_columns].values[0])) + \
                self.data_min[self.label_columns].values[0]
        # return ( y * self.data_std[self.label_columns].values[0]) + \
        #         self.data_mean[self.label_columns].values[0]

    @property
    def train(self):
        return self.create_dataset(self.train_df, self.aux1, self.aux2)#, self.batch_size)

    @property
    def val(self):
        return self.create_dataset(self.val_df, self.aux1, self.aux2)

    @property
    def test(self):
        return self.create_dataset(self.test_df, self.aux1, self.aux2)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    @property
    def labels(self):
        return self.data[self.label_columns]
#%%
# def creat_dataset(data, n_in = 21, n_out = 7):pg
#     # [samples, timesteps, features]
#     X = np.empty(shape=(0,n_in,data.shape[-1]))
#     Y = np.empty(shape=(0,n_out,2))
#     for i in range(len(data) - (n_in + n_out)):
#         x = data[i:(i+n_in)].values
#         y = data.loc[(i+n_in):(i+n_in+n_out-1),["Date","Maximum_Power_This_Year"]].values
#         X = np.vstack([X,np.array(x)[np.newaxis,...]])
#         Y = np.vstack([Y,np.array(y)[np.newaxis,...]])
#     # X = np.array(X)
#     # Y = np.array(Y)
#     return X, Y

# def create_dataset_holiday(data, n_in = 21, n_out = 7):
#     X = np.empty(shape=(0,n_out,2))
#     for i in range(len(data) - (n_in + n_out)):
#         x = data.loc[(i+n_in):(i+n_in+n_out-1),["weekday","holiday"]].values
#         X = np.vstack([X, np.array(x)[np.newaxis,...]])
#     return X

# def split_data(X, Y = None, test_size = 14):
#     if Y is None :
#         X_train, X_test = X[:(len(X)-test_size)], X[(len(X)-test_size):]
#         print("X_train shape : {}\n X_test shape : {}".format(X_train.shape, X_test.shape))
#         return X_train, X_test
#     else :
#         X_train, X_test = X[:(len(X)-test_size)], X[(len(X)-test_size):]
#         y_train, y_test = Y[:(len(Y)-test_size)], Y[(len(Y)-test_size):]
#         print("X_train shape : {}, y_train shape : {}\nX_test shape : {}, y_test shape : {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
#         # train = np.array(np.split(train, len(train)/7))
#         # test = np.array(np.split(test, len(test)/7))
#         return X_train, y_train, X_test, y_test


# %%
