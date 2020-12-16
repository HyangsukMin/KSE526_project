#%%
import pandas as pd
import glob
import os
pd.set_option('display.max_columns', 500)
data_dir = "./data/KPX/"
# fname = os.listdir(os.path.join(data_dir,"Load"))
#%%
################################
# KPX Load Dataset
################################
fnames = glob.glob(os.path.join(data_dir,"load") + "/*.xls")
df = pd.DataFrame()
for fname in fnames:
    tmp = pd.read_excel(fname, header = 1)
    df = df.append(tmp,ignore_index=True)

# target = Maximum_Power_This_Year
df.columns = ["No","Date","Installed_Capacity","Supply_Capacity","Maximum_Power_Last_Year","Maximum_Power_This_Year","Increasment","Supply_Reserve","Reserve_Rate"]
df.sort_values("No",inplace=True)
df.reset_index(drop=True, inplace=True)

df["Timestamp"] = df["Date"].str[-5:]
df["Date"] = df["Date"].str[:-6]
df.loc[df.Timestamp == "24:00","Timestamp"] = "00:00"

df.iloc[:,[2,3,4,5,7]] = df.iloc[:,[2,3,4,5,7]].apply(lambda x : x.str.replace(",",""))

df.loc[df.Installed_Capacity == "7.367.3","Installed_Capacity"] = 7367.3
df.loc[df.Installed_Capacity == "8.159.3","Installed_Capacity"] = 8159.3
df.loc[df.Installed_Capacity == "86.33.3","Installed_Capacity"] = 8633.3
df.loc[df.Supply_Capacity == "6740..3","Supply_Capacity"] = 6740.3
df.loc[df.Supply_Capacity == "8.054.1","Supply_Capacity"] = 8054.1
df.loc[df.Maximum_Power_Last_Year == "5.345.3","Maximum_Power_Last_Year"] = 5345.3
df.loc[df.Maximum_Power_Last_Year == "5.890.1","Maximum_Power_Last_Year"] = 5890.1
df.loc[df.Maximum_Power_Last_Year == "5.425.3","Maximum_Power_Last_Year"] = 5425.3
df.loc[df.Maximum_Power_Last_Year == "6.467.6","Maximum_Power_Last_Year"] = 6467.6
df.loc[df.Date == "2020.11.14","Maximum_Power_Last_Year  "] = 6042.1 #60421.0
df.loc[df.Maximum_Power_This_Year == "5.307.4","Maximum_Power_This_Year"] = 5307.4
df.loc[df.Supply_Reserve == "753..2","Supply_Reserve"] = 753.2
df.loc[df.Supply_Reserve == "772..9","Supply_Reserve"] = 772.9

df.iloc[:,[2,3,4,5,7]] = df.iloc[:,[2,3,4,5,7]].apply(lambda x : x.astype(float))
df.to_csv("./data/preprocess/KPX_load.csv", index=False)


# %%
################################
# KPX SMP Dataset
################################
# fnames = glob.glob(os.path.join(data_dir,"SMP") + "/*.xls")
# df = pd.DataFrame()
# for fname in fnames:
#     tmp = pd.read_excel(fname, header = 3)
#     df = df.append(tmp,ignore_index=True)
# df1 = pd.melt(df, id_vars="구분", value_vars=df.columns[1:-3])
# df2 = pd.melt(df, id_vars="구분", value_vars=df.columns[-3:])

# df1.columns = ["Date","Times","SMP"]
# df2.columns = ["Date","Type","SMP"]
# df1.to_csv("./data/preprocess/KPX_SMP_hourly.csv",index=False)
# df2.to_csv("./data/preprocess/KPX_SMP_mean.csv",index=False)
# %%
################################
# Meteorolgy Dataset
################################
data_dir = "./data/Meteorolgy/"
fnames = glob.glob(data_dir + "/OBS_ASOS_DD_*.csv")
df = pd.DataFrame()
for fname in fnames[:]:
    tmp = pd.read_csv(fname, header = 0, encoding='cp949')
    df = df.append(tmp,ignore_index=True)
df.columns = ["location","location_nm","Date", # 3
                "avg_temp","min_temp","min_temp_hhmi","max_temp","max_temp_hhmi", # 5
                "rain_duration_hr","max_rain_10mi","max_rain_10mi_hhmi","max_rain_1h","max_rain_1h_hhmi","rain_day", # 6
                "max_wind_sec","max_wind_sec_direction","max_wind_sec_hhmi","max_wind","max_wind_direction","max_wind_hhmi","avg_wind","pungjeonghap","freq_wind_direction", # 9
                "avg_dew_point","min_relative_humidity","min_relative_humidity_hhmi","avg_relative_humidity", # 4
                "avg_steam_pressure","avg_spot_pressure", # 2
                "max_sea_pressure","max_sea_pressure_hhmi","min_sea_pressure","min_sea_pressure_hhmi","avg_sea_pressure", # 5
                "sunshine_hr","bright_sunshine_hr", # 2
                "none1","none2","none3","none4","none5","none6","none7","none8","none9","none10", # 10
                "avg_land_temp","min_grass_temp", # 2
                "none11","none12","none13","none14","none15","none16","none17","none18","none19","none20",
                "none21","none22","none23","none24" # 14
                ]
df["Date"] = df["Date"].str.replace("-",".")
df.sort_values(["location","location_nm","Date"],inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv("./data/preprocess/Meteorology.csv", index=False)
# %%
################################
# Oil Dataset
################################
# df = pd.read_excel("./data/Others/ShellPrice.xlsx")
df = pd.read_csv("./data/Others/ShellPrice.csv",encoding='cp949')
df.columns = ["Date","gasoline1","gasoline2","diesel","kerosene"]
df["Date"] = df["Date"].str.replace(". ", ".")
df.to_csv("./data/preprocess/shell_price.csv", index=False)
# %%
################################
# Exchange Dataset
################################
df = pd.read_csv("./data/Others/Exchange.csv")
df["Date"] = df["Date"].str.replace(". ", ".")
df.iloc[:,1] = df.iloc[:,1].str.replace(",","").astype(float)
df.iloc[:,2] = df.iloc[:,2].str.replace(",","").astype(float)
df.iloc[:,3] = df.iloc[:,3].str.replace(",","").astype(float)
df.iloc[:,4] = df.iloc[:,4].str.replace(",","").astype(float)
df.iloc[:,5] = df.iloc[:,5].str.replace("%","").astype(float)

date = pd.date_range('2020.01.01', end='2020.12.10', freq='d')
date = pd.DataFrame(columns=["Date"],data=date.astype(str).values)
date["Date"] = date["Date"].str.replace("-",".")

df = pd.merge(date, df,on="Date",how='left')
def fill_weekend(df):
    for i in range(len(df)):
        if pd.isna(df.iloc[i,1]) :
            df.iloc[i,1:] = df.iloc[(i-1),1:]
    return df
df = fill_weekend(df)
df.columns = [["Date","Last","Open","Hihg","Low","Rate"]]
df.to_csv("./data/preprocess/exchange.csv", index=False)
# %%
