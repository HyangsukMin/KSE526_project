#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from workalendar.asia import SouthKorea # 한국의 공휴일, version : 1.1.1

# tf.debugging.set_log_device_placement(True)
# filelist = [
#     "./data/COVID/covid19.csv",
#     "./data/preprocess/KFX_load.csv",
#     "./data/preprocess/Meteorology.csv"
# ]
#%%
class WindowGenerator():
    def __init__(self, input_width = 7, label_width = 7, shift = 14, label_columns = ["Maximum_Power_This_Year"], features = None):

        ##############################################################
        # Raw data
        ##############################################################
        kpx_load = pd.read_csv("./data/preprocess/KPX_load.csv")
        self.data = kpx_load[["Date","Installed_Capacity","Supply_Capacity","Maximum_Power_Last_Year","Maximum_Power_This_Year","Supply_Reserve"]]

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
                # meteorology.columns = ["location","Date",
                #                         "avg_temp","min_temp","max_temp", # 평균기온
                #                         "max_rain_1h", #강우량
                #                         "avg_dew_point", #이슬점
                #                         "avg_relative_humidity", #상대습도
                #                         "sunshine_hr", #일조시간
                #                         "avg_land_temp", #지면온도
                #                         ]
                # meteorology = meteorology.fillna(0).groupby("Date").agg('mean')
                self.data = pd.merge(self.data, meteorology, on="Date")
            if "covid" in features:
                """
                """
                covid = pd.read_csv("./data/COVID/covid19.csv")
                covid["Date"] = covid["Date"].str.replace(" ","")
                covid["Sum_diff"] = np.gradient(covid.Sum,1)
                covid["Sum_diff2"] = np.gradient(covid.Sum,2)
                covid = covid[["Date",
                                "Sum_diff2", #전일대비 증가량의 증가량
                                "Sum_diff" #전일대비 증가량
                ]]
                self.data = pd.merge(self.data, covid, on="Date")

            if "gas" in features:
                """
                """
                gas = pd.read_csv("./data/preprocess/shell_price.csv")
                gas["gasoline_diff"] = np.gradient(gas.gasoline2,1)
                gas["diesel_diff"] = np.gradient(gas.diesel,1)
                gas = gas[["Date",
                            "gasoline2", #일반휘발유 가격
                            "diesel", #경유 가격
                            "gasoline_diff", #일반휘발유 가격 전일대비 증가량
                            "diesel_diff" #경유 전일대비 증가량
                ]]
                self.data = pd.merge(self.data, gas, on="Date")

            if "exchange" in features:
                """
                """
                exchange = pd.read_csv("./data/preprocess/exchange.csv")
                exchange["Last_diff"] = np.gradient(exchange.Last,1)
                exchange = exchange[["Date",
                                    "Last", #종가
                                    "Last_diff" #종가 전일 대비 증가량
                ]] 
                self.data = pd.merge(self.data, exchange, on="Date")
        
        """
        Others : holiday, weekday information
        """
        date = pd.date_range('2020.01.01', end='2020.11.24', freq='d')
        date = pd.DataFrame(columns=["Date"],data=date.astype(str).values)
        date["Date"] = date["Date"].str.replace("-",".")
        date["weekday"] = pd.to_datetime(date["Date"]).dt.weekday
        week_dict = {0:1,1:0,2:0,3:0,4:0,5:1,6:1}
        date["weekday"] = date["weekday"].map(week_dict)
        date["holiday"] = 0
        date.loc[date.Date.isin(pd.Series(np.array(SouthKorea().holidays(2020))[:, 0]).map(str).str.replace("-",".")),"holiday"] = 1
        
        self.data = pd.merge(self.data, date, on="Date", how='left')
        
        self.data.fillna(0,inplace=True)
        
        self.train_df = self.data[self.data.Date < "2020.11.01"]
        self.val_df = self.data[self.data.Date >= "2020.11.01"]

        self.label_columns = label_columns
        ##############################################################
        # Work out the label column indices.
        ##############################################################
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(self.train_df.columns)}

        ##############################################################
        # Work out the window parameters
        ##############################################################
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = self.input_width + self.shift

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

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    def make_dataset(self, data):
        data = data.drop(["Date"],axis=1)
        data = np.array(data, dtype=np.float64)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

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


