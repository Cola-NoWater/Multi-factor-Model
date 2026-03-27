import baostock as bs
import pandas as pd
import tushare as ts
import time as ti
df = pd.read_csv("/home/linyuchang/wcp/model/raw_data/src/month_map.csv")
date_list = df["trade_date"].tolist()
date_list = [date for date in date_list if (date>="2000-01-01") & (date<="2020-12-31")]
pro = ts.pro_api('23166cc175fbd78f1917350a8eac8ea2d534e8458c1f6bd43d0ff8e5')
data = pd.DataFrame()
for date in date_list:
    date = date.replace("-","")
    print(date)
    ti.sleep(0.4)
    df = pro.index_weight(index_code='399300.SZ',trade_date=date)
    if not df.empty:
        data = pd.concat([data,df],ignore_index=True)
data.to_csv("/home/linyuchang/wcp/model/MFIE_Model/test.csv", encoding="utf-8", index=False)
print("数据拿取成功")
if __name__ == "__main__":
    df = pd.read_csv("/home/linyuchang/wcp/model/MFIE_Model/test.csv")
    data = df.pivot_table(index="con_code",columns="trade_date",values="weight")
    data.to_csv("/home/linyuchang/wcp/model/raw_data/src/hs300.csv")
    df = pd.read_csv("/home/linyuchang/wcp/model/raw_data/src/hs300.csv")