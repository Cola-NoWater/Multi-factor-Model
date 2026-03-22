import baostock as bs
import pandas as pd

datelist = pd.read_csv("/home/linyuchang/wcp/model/raw_data/src/month_map.csv")
datelist
date =  [x for x in datelist["calendar_date"].tolist() if (x<="2020-12-31") & (x>="2000-01-01") ]
stock_list = pd.read_csv("/home/linyuchang/wcp/model/raw_data/src/all_stocks.csv",encoding="gb18030")
data = pd.DataFrame()
# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)
# 获取行业分类数据
for day in date:
    rs = bs.query_stock_industry(date=day)
# rs = bs.query_stock_basic(code_name="浦发银行")
    print('query_stock_industry error_code:'+rs.error_code)
    print('query_stock_industry respond  error_msg:'+rs.error_msg)
# 打印结果集
    industry_list = []
    while (rs.error_code == '0') & rs.next():
# 获取一条记录，将记录合并在一起
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    result["updateDate"] = day
    if len(result)>0:
        result["code"] = result["code"].str[3:] +"."+ result["code"].str[0:2].str.upper()
        result = result[["updateDate","code","industry"]]
        data = pd.concat([data,result])
data=data.pivot_table(index="code",columns="updateDate",values="industry",aggfunc="first")
# 结果集输出到csv文件
data
data.to_csv("/home/linyuchang/wcp/model/raw_data/industry_citic.csv", encoding="utf-8")


if __name__ == '__main__':
    df = pd.read_csv("/home/linyuchang/wcp/model/raw_data/industry_citic.csv",index_col="code")
    df.columns = pd.to_datetime(df.columns)
    df = df.set_index("code")  
    df.to_csv("/home/linyuchang/wcp/model/raw_data/industry_citic.csv", encoding="utf-8",index=True)