#%%
import requests
from bs4 import BeautifulSoup as bs
req = requests.get("http://www.happydr.co.kr/?act=&mid=CurrentSupplyView&vid=&view=table&option_date=2018-01-01")
html = req.text
soup = bs(html, 'html.parser')
table = soup.select("#dataView > div:nth-child(3) > table > tbody > tr")
table[0]
# %%
import pandas as pd
url = "http://www.happydr.co.kr/?act=&mid=CurrentSupplyView&vid=&view=table&option_date={}"
date = pd.date_range(start="2018-01-01",end="2020-11-25",freq="d")
time = list()
supply = list()
load = list()
supply_capacity = list()
operational_capacity = list()
operational_capacity_per = list()
for d in date:
    print(d,end="\t")
    req = requests.get(url.format(d))
    html = req.text
    soup = bs(html,'html.parser')
    table = soup.select("#dataView > div:nth-child(3) > table > tbody > tr")
    for t in table:
        t = t.text.split("\n")[1:-1]
        time.append(t[0])
        supply.append(t[1])
        load.append(t[2])
        supply_capacity.append(t[3])
        operational_capacity.append(t[4])
        operational_capacity_per.append(t[5])
# %%
df = pd.DataFrame(
    {"Date":time,
    "supply_capacity":supply,
    "load":load,
    "supply_reserve":supply_capacity,
    "operational_resesrve":operational_capacity,
    "operational_reserve_per":operational_capacity_per}
)

# %%
# df.to_csv("./data/real-time_load.csv",index=False)
# %%

df.iloc[:,1] = df.iloc[:,1].str.replace(",","").astype(float)/1000
df.iloc[:,2] = df.iloc[:,2].str.replace(",","").astype(float)/1000
df.iloc[:,3] = df.iloc[:,3].str.replace(",","").astype(float)/1000
df.iloc[:,4] = df.iloc[:,4].str.replace(",","").astype(float)/1000
df.iloc[:,5] = df.iloc[:,5].str.replace("%","").astype(float)

#%%
df.sort_values("Date",inplace=True)
df.reset_index(drop=True,inplace=True)
# %%
df.to_csv("./data/preprocess/realtime_load.csv",index=False)
# %%
