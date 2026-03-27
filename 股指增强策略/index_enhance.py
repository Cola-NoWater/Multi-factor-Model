
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from functools import reduce
import lightgbm as lgb

from collections import defaultdict
from copy import deepcopy

from scipy.optimize import linprog
from factor_generate import FactorGenerater
from factor_preprocess import info_cols
from single_factor_test import *
from pandas.tseries.offsets import MonthEnd

__all__ = ['index_enhance_model', 'get_factor', 'get_stock_wt_in_index', 'factor_process']

sns.set_style(style="darkgrid")

plt.rcParams['font.sans-serif'] = ['SimHei']  #正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    #正常显示负号
plt.rcParams['figure.figsize'] = (16.0, 9.0)  #图片尺寸设定（宽 * 高 cm^2)
plt.rcParams['font.size'] = 15                #字体大小

#工作目录，存放代码
work_dir = os.path.dirname(__file__)
#经过预处理后的因子截面数据存放目录
factor_panel_path = os.path.join(work_dir, '因子预处理模块', '因子（已预处理）')
#原始经过预处理的因子的矩阵数据存放目录
factor_matrix_path = os.path.join(work_dir, '因子预处理模块', "矩阵数据",'因子矩阵')
#合成、正交因子存放目录（如无则自动生成）
rm_save_path = os.path.join(work_dir, '收益模型')
#测试结果图表存放目录（如无则自动生成）
index_enhance_dir = os.path.join(work_dir, '指数增强模型')

industry_benchmark = ''      #中信一级行业
industry_list = []


#自动生成合成、正交因子存放目录
if not os.path.exists(rm_save_path):
    os.mkdir(rm_save_path)
#自动生成指数增强模型结果存放目录
if not os.path.exists(index_enhance_dir):
    os.mkdir(index_enhance_dir)

factor_generater = FactorGenerater()

def get_stock_wt_in_index(index):
    """
    获取指数（000300.SH或000905.SH）中各截面期成分股所占权重
    """
    global factor_generater
    if index.startswith('000300') or index.startswith('399300'):
        index_wt = factor_generater.hs300_wt
    elif index.startswith('000905') or index.startswith('399905'):
        index_wt = factor_generater.zz500_wt
    else:
        msg = f'暂不支持当前指数：{index}'
        raise Exception(msg)
    return index_wt
#市场状况
def get_market_status():
    """
    输入：沪深300日线数据
    输出：'good' / 'bad'
    """
    # 计算60日收益率
    hs300_df = pd.read_csv("/home/linyuchang/wcp/model/raw_data/__temp_index_daily__/000300.SH.csv",index_col=["trade_date"])
    hs300_df.index = hs300_df.index.astype("str")
    hs300_df = hs300_df.sort_index()
    hs300_df["mom_6"] = hs300_df['close']/ hs300_df['close'].rolling(6).mean() - 1
    hs300_df["ma3"] = hs300_df["close"].rolling(3).mean()
    hs300_df["ma3_up"] = hs300_df["ma3"] > hs300_df["ma3"].shift(1)  # 均线向上
    def judge_status(row):
        if row["mom_6"] > 0 and row["ma3_up"]:
            return "good"
        else:
            return "bad"

    hs300_df["status"] = hs300_df.apply(judge_status,axis=1)
    return hs300_df["status"] 

    

def get_factor_corr(factors=None, codes=None):
    """
    计算因子相关系数
    """
    if factors is None:
        factors = get_factor_names()
    factors_matrix_dat = get_factor(factors)  ##获取指定因子全部数据
    factors_panel_dat = concat_factors_panel(factors, factors_matrix_dat, codes, False, False)  #将一个或者多个因子矩阵数据转换为因子截面数据,按需可以加入行业伪变量和市值对数
    corrs = []
    for date in sorted(factors_panel_dat.keys()):
        factor_panel = factors_panel_dat[date]
        corrs.append(factor_panel.corr())

    avg_corr = reduce(lambda df1, df2: df1 + df2, corrs) / len(corrs)  ##累加求均值
    return avg_corr

def factor_concat(factors_to_concat, new_factor_name, weight=None):
    """
    因子合成：
    输入：待合并因子的名称(,分隔); 合成后的因子存储名称(自动添加_con后缀); 合成权重(默认等权)
    输出：合成后因子的因子截面数据和矩阵数据
    """
    global factor_panel_path, rm_save_path, info_cols
    if not new_factor_name.endswith('con'):
        new_factor_name += '_con'
    cfactor_spath = os.path.join(rm_save_path, '新合成因子')
    cpanel_spath = os.path.join(cfactor_spath, '因子截面')
    cmatrix_spath = os.path.join(cfactor_spath, '因子矩阵')
    if not os.path.exists(cfactor_spath):
        os.mkdir(cfactor_spath)
        os.mkdir(cpanel_spath)
        os.mkdir(cmatrix_spath)

    if ',' in factors_to_concat:
        factors_to_concat = factors_to_concat.split(',')

    if weight is None:
        apply_func = np.mean
        col_name = new_factor_name+'_equal'
    else:
        apply_func = lambda df: np.sum(weight*df)
        col_name = new_factor_name

    if os.path.exists(os.path.join(cmatrix_spath, col_name+'.csv')):
        print(f'{col_name}因子数据已存在')
        return

    panelfactors = os.listdir(cpanel_spath)

    for f in os.listdir(factor_panel_path):
        dat = pd.read_csv(os.path.join(factor_panel_path, f), encoding="utf-8", engine='python', index_col=[0])
        factor_dat = dat[factors_to_concat]
        factor_concated = factor_dat.apply(apply_func, axis=1)
        factor_concated.name = col_name
        if panelfactors: #判断目标文件是否存在,存在就只需更新内容,故意不写成 if f in panelfactors,为了就是能提早发现错误
            panel_dat = pd.read_csv(os.path.join(cpanel_spath, f), encoding="utf-8", engine='python', index_col=[0])
            if col_name in panel_dat.columns:
                del panel_dat[col_name]
            panel_dat = pd.concat([panel_dat, factor_concated], axis=1)
        else:
            panel_dat = pd.concat([dat[info_cols], factor_concated], axis=1)

        panel_dat.to_csv(os.path.join(cpanel_spath, f), encoding="utf-8")

    panel_to_matrix([col_name], factor_path=cpanel_spath, save_path=cmatrix_spath) #将刚生成的截面数据转换成矩阵数据存放到cmatrix_spath目录当中
    print(f"创建{col_name}因子数据成功.")

def orthogonalize(factors_y, factors_x, codes=None, index_wt=None):
    """
    因子正交：
    输入：因变量(y)、自变量(x)因子名称（,分隔），类型：字符串
    输出：经过正交的因子截面数据和因子矩阵数据
    """
    global rm_save_path, factor_panel_path, info_cols
    ofactor_spath = os.path.join(rm_save_path, '正交后因子')
    opanel_spath = os.path.join(ofactor_spath, '因子截面')
    omatrix_spath = os.path.join(ofactor_spath, '因子矩阵')
    if not os.path.exists(ofactor_spath):
        os.mkdir(ofactor_spath)
        os.mkdir(opanel_spath)
        os.mkdir(omatrix_spath)

    for fac in factors_y.copy():
        if os.path.exists(os.path.join(omatrix_spath, fac+'_ortho.csv')):
            print(f'{fac}_ortho因子数据已存在')
            factors_y.remove(fac)

    if len(factors_y) == 0:
        return

    panel_y = concat_factors_panel(factors_y, codes=codes, ind=True, mktcap=True)
    panel_x = concat_factors_panel(factors_x, codes=codes, ind=True, mktcap=True)

    ortho_y = {}
    for date in sorted(panel_x.keys()):
        y = panel_y[date]
        X = panel_x[date]
        date_wt = date.strftime("%Y%m%d")
        #cur_index_wt = index_wt[date_wt].dropna()
        data_to_regress = pd.concat([X, y], axis=1)
        #mut_index = data_to_regress.index.intersection(cur_index_wt.index)
        #data_to_regress = data_to_regress.loc[mut_index, :]
        data_to_regress = data_to_regress.dropna(how='any', axis=0)
        cut_loc = len(y.columns) ##y的列数
        X, ys = data_to_regress.iloc[:, :-cut_loc], data_to_regress.iloc[:, -cut_loc:]
        resids = pd.DataFrame()
        #params_a = pd.DataFrame()
        for fac in ys.columns:
            y = ys[fac]

            _, params, resid_y = regress(y, X, intercept=True)
            #params_a = pd.concat([params_a, params], axis=1)
            resid_y.name = fac + '_ortho'
            resids = pd.concat([resids, resid_y], axis=1)
        ortho_y[date] = resids

    for date in ortho_y.keys():
        date_str = str(date)[:10]
        cur_panel_ortho = ortho_y[date]
        basic_info = pd.read_csv(os.path.join(factor_panel_path, date_str+'.csv'), encoding="utf-8", engine='python', index_col=[0])[info_cols]
        new_panel = pd.merge(basic_info, cur_panel_ortho, left_on='code', right_index=True)
        new_panel.to_csv(os.path.join(opanel_spath, date_str+'.csv'), encoding="utf-8")

    factors_ortho = [fac+'_ortho' for fac in factors_y]
    panel_to_matrix(factors_ortho, factor_path=opanel_spath, save_path=omatrix_spath) #将刚生成的截面数据转换成矩阵数据存放到omatrix_spath目录当中
    print(f"创建{','.join(factors_ortho)}因子数据成功.")

def get_panel_data(names, fpath, codes=None):  ##数据存到字典 因子名称：（code-时间）  读取截面数据
    res = defaultdict(pd.DataFrame)
    if not isinstance(names, list):
        names = [names]
    for file in os.listdir(fpath):
        date = pd.to_datetime(file.split('.')[0])
        datdf = pd.read_csv(os.path.join(fpath, file), encoding="utf-8", engine='python', index_col=['code'])
        for name in names:
            dat = datdf.loc[:, name]
            dat.name = date
            if codes is not None:
                dat = dat.loc[codes]
            res[name] = pd.concat([res[name], dat], axis=1)
    return res

def get_matrix_data(name, fpath, codes=None): ##读取单因子多日期多股票数据
    data = pd.read_csv(os.path.join(fpath, name+'.csv'), encoding="utf-8", engine='python', index_col=[0])
    data.columns = pd.to_datetime(data.columns)

    if codes is not None:
        codes = codes.intersection(data.index)
        data = data.loc[codes, :]
    return {name: data}

def get_factor(factor_names, codes=None):
    """
    获取指定因子全部数据（仅预处理、合成、正交）
    """
    #指定因子所在因子路径
    factor_paths = [(f, get_factor_path(f)) for f in factor_names]
    #矩阵形式保存因子的路径
    factors_matrix = {fname: path for fname, path in factor_paths if path.endswith('因子矩阵')}
    #截面形式保存因子的路径
    factors_panel = defaultdict(list)
    for fname, path in factor_paths:
        if path.endswith('截面') or '已预处理' in path:
            factors_panel[path].append(fname)
    #读取矩阵形式保存的因子
    res = {}
    for fname, fpath in factors_matrix.items():
        res.update(get_matrix_data(fname, fpath, codes))
    #读取截面形式保存的因子
    for fpath, fnames in factors_panel.items():
        res.update(get_panel_data(fnames, fpath, codes))
    return res #{因子名: dataframe(因子矩阵数据)}

def get_factor_path(factor_name, frame='matrix'):  ##识别路径
    """
    根据因子名称后缀，识别因子路径（仅预处理、合成、正交）
    """
    global factor_panel_path, rm_save_path, factor_matrix_path, info_cols, industry_benchmark
    new_concated_spath = os.path.join(rm_save_path, '新合成因子')
    orthoed_spath = os.path.join(rm_save_path, '正交后因子')

    basic_infos = [name for name in info_cols if name not in ('MKT_CAP_FLOAT', f'industry{industry_benchmark}', 'PCT_CHG_NM')]
    if factor_name in basic_infos:
        return factor_panel_path

    if factor_name.endswith('_con') or factor_name.endswith('_con_equal'):
        new_concated = True
    else:
        new_concated = False
        if factor_name.endswith('_ortho'):
            orthoed = True
        else:
            orthoed = False

    if frame == 'panel':
        if new_concated:
            open_path = os.path.join(new_concated_spath, '因子截面')
        elif orthoed:
            open_path = os.path.join(orthoed_spath, '因子截面')
        else:
            open_path = factor_panel_path

    elif frame == 'matrix':
        if new_concated:
            open_path = os.path.join(new_concated_spath, '因子矩阵')
        elif orthoed:
            open_path = os.path.join(orthoed_spath, '因子矩阵')
        else:
            open_path = factor_matrix_path
    else:
        raise TypeError(f"不支持的因子数据格式：{frame}")
    return open_path

def concat_factors_panel(factors=None, factors_dict=None, codes=None, ind=True, mktcap=True):
    """
    将一个或者多个因子矩阵数据转换为因子截面数据,按需可以加入行业伪变量和市值对数
    """
    global industry_benchmark,industry_list
    factors = deepcopy(factors)
    if factors:
        if isinstance(factors, str):
            factors = factors.split(',')
    else:
        factors = []

    if ind:
        factors.append(f'industry{industry_benchmark}')
    if mktcap:
        factors.append('MKT_CAP_FLOAT')

    if codes is not None and factors_dict is not None:
        for _ , datdf in factors_dict.items():
            codes = codes.intersection(datdf.index)
        factors_dict = {fac: datdf.loc[codes,:] for fac, datdf in factors_dict.items()}
    if (factors_dict is None) or ('MKT_CAP_FLOAT' in factors) or (f'industry{industry_benchmark}' in factors):  ##行业和市值是基础列，存储路径和其他有所不同
        matrix = {}
        for fac in factors:
            fpath = get_factor_path(fac)
            matrix.update(get_matrix_data(fac, fpath, codes))
        if factors_dict:
            matrix.update(factors_dict)
    else:
        matrix = factors_dict


    panel = defaultdict(pd.DataFrame)
    industry_list.clear()
    #对每个时间截面，合并因子数据
    facs = sorted(matrix.keys())

    for fac in facs:
        for date in matrix[fac]:
            cur_fac_panel_data = matrix[fac][date]
            cur_fac_panel_data.name = fac
            if ('industry' == fac) and (ind == True):
                industry_list.extend(cur_fac_panel_data)
                cur_fac_panel_data = pd.get_dummies(cur_fac_panel_data).astype("int")
            elif fac == 'MKT_CAP_FLOAT' and (mktcap == True):
                cur_fac_panel_data = np.log(cur_fac_panel_data)
                cur_fac_panel_data.name = 'ln_mkt_cap'

            panel[date] = pd.concat([panel[date], cur_fac_panel_data], axis=1)  
    industry_list = list(set(industry_list))
    industry_list = [x for x in industry_list if pd.notna(x)]   
    return panel

def get_exponential_weights(window=12, half_life=6):  #计算衰减指数
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] 

def wt_sum(series, wt):
    if len(series) < len(wt):
        return np.sum(series * wt[:len(series)] / np.sum(wt[:len(series)]))
    else:
        return np.sum(series * wt / np.sum(wt))

def factor_return_forecast(factors_x, factor_data=None, window=12, half_life=6):
    """
    因子收益预测：
    输入：自变量(x)因子名称（,分隔），类型：字符串
    输出：截面回归得到的因子收益率预测值，行：因子名称，列：截面回归当期日期
    """
    index_wt = get_stock_wt_in_index('000300.SH')
    ret_matrix = get_factor(['PCT_CHG_NM'])['PCT_CHG_NM']

    if factor_data is None:
        panel_x = concat_factors_panel(factors_x)
    else:
        panel_x = factor_data

    #逐期进行截面回归，获取回归系数，作为因子收益   因子--->因子收益率
    factor_rets = pd.DataFrame()
    for date in sorted(panel_x.keys()):
        y = ret_matrix[date] #下期(期末)股票收益率
        X = panel_x[date]    #当期(期末)因子值(因子暴露,因子载荷),也就是下期期初的值
        date_str = date.strftime("%Y%m%d")
        cur_index_wt = index_wt[date_str].dropna()

        data_to_regress = pd.concat([X, y], axis=1)
        data_to_regress = data_to_regress.dropna(how='any', axis=0)
        mut_index = data_to_regress.index.intersection(cur_index_wt.index)
        data_to_regress = data_to_regress.loc[mut_index, :]
        X, y = data_to_regress.iloc[:, :-1], data_to_regress.iloc[:, -1]
        for fac in X.sum()[X.sum() == 0].index:
            if fac not in factors_x:
                del X[fac]
        w = X['ln_mkt_cap']
        _, cur_factor_ret, _ = regress(y, X, w) #重点: 回归出来是下期因子实际收益率
        cur_factor_ret.name = date              #实际内容是指date期的下一期因子实际收益率
        factor_rets = pd.concat([factor_rets, cur_factor_ret], axis=1)

    #因子实际收益率,行列转换一下,方便接下来处理
    factor_rets = factor_rets.T

    #对ROE_q以及growth因子的负值纠正为0
    for fac in ['ROE_q', 'growth']:
        try:
            fac_name = [f for f in factor_rets.columns if f.startswith(fac)][0]
        except IndexError:
            continue
        factor_rets[fac_name] = factor_rets[fac_name].where(factor_rets[fac_name] >= 0, 0)

    #利用历史[因子实际收益率]预测[因子预测收益率],当然预测方法可以有很多,这里只演示了两种
    if half_life:
        exp_wt = get_exponential_weights(window=window, half_life=half_life) #指数加权权重
        factor_rets = factor_rets.rolling(window=window).apply(wt_sum, args=(exp_wt,))###
    else:
        factor_rets = factor_rets.rolling(window=window).mean().shift(1)
    factor_rets = factor_rets.dropna(how='all', axis=0)
    #重点1: 每个单元存放的都是相应下期因子预测收益率
    #重点2: [因子预测收益率序列]长度要比前面的[因子实际收益率序列]少window期
    return factor_rets

def get_est_stock_return(factors, factors_panel, est_factor_rets, window=12, half_life=6):
    """
    根据之前预测的[因子预测收益率],计算得到各股票的截面预期(预测)收益
    """
    est_stock_rets = pd.DataFrame()
    for date in est_factor_rets.index:
        cur_factor_panel = factors_panel[date] #date期因子值(因子暴露,因子载荷)
        cur_factor_panel = cur_factor_panel[factors]
        cur_factor_panel = cur_factor_panel.dropna(how='any', axis=0)
        cur_est_stock_rets = np.dot(cur_factor_panel, est_factor_rets.loc[date]) #参数: date期因子值, date+1期的[因子预测收益率]
        #实际内容是date+1期的股票预期收益率
        cur_est_stock_rets = pd.DataFrame(cur_est_stock_rets, index=cur_factor_panel.index, columns=[date])
        est_stock_rets = pd.concat([est_stock_rets, cur_est_stock_rets], axis=1)
    return est_stock_rets #重点:每个存储单元包含的是对应日期的下一期的股票预期收益率

def get_refresh_days(tradedays, start_date, end_date):
    """
    获取调仓日期（回测期内的每个月首个交易日）
    """
    tdays = tradedays
    sindex = get_date_idx(tradedays, start_date)
    eindex = get_date_idx(tradedays, end_date)
    tdays = tdays[sindex:eindex+1]
    return (nd for td, nd in zip(tdays[:-1], tdays[1:]) 
            if td.month != nd.month)

def get_date_idx(tradedays, date):
    """
    返回传入的交易日对应在全部交易日列表中的下标索引
    """
    datelist = list(tradedays)
    date = pd.to_datetime(date)
    try:
        idx = datelist.index(date)
    except ValueError:
        datelist.append(date)
        datelist.sort()
        idx = datelist.index(date)
        if idx == 0:
            return idx + 1
        else:
            return idx - 1
    return idx

def plot_net_value(records, benchmark, method_name, save_path, start_date, end_date):
    """
    绘制回测净值曲线
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    records = records[['benchmark_nv', 'net_value']]
    records /= records.iloc[0,:]
    plt.plot(records)
    plt.legend([benchmark, method_name], loc=2)
    plt.title('回测净值')
    plt.savefig(os.path.join(save_path, f'{method_name}_{start_date}-{end_date}.jpg'))
    plt.close()

def lp_solve(cur_est_rets, limit_factors, cur_benchmark_wt, A_sector,b_sector,num_multi=5):
    """
    线性规划计算函数：
    输入：截面预期收益，约束条件（风险因子），截面标的指数成分股权重，个股权重约束倍数
    输出：经优化后的组合内个股权重
    """
    data = pd.concat([cur_est_rets, limit_factors, cur_benchmark_wt], axis=1)
    data = data.dropna(how='any', axis=0)
    cur_est_rets, limit_factors, cur_benchmark_wt = (data.iloc[:, 0:1], data.iloc[:, 1:-1], data.iloc[:, -1])
    com_code = cur_benchmark_wt.index
    cur_benchmark_wt = cur_benchmark_wt / cur_benchmark_wt.sum()

    c = cur_est_rets.values.flatten()##展平成一维
    
    industry_rolling = False
    if not industry_rolling:
        A_sector = None
        b_sector = None    
    if A_sector is None or A_sector.empty:
        A_ub = A_sector
        b_ub = b_sector
        A_eq = np.r_[limit_factors.T.values, np.repeat(1, len(limit_factors)).reshape(1, -1)]  #np.r_  拼接  #等式约束
        b_eq = np.r_[np.dot(limit_factors.T, cur_benchmark_wt), np.array([1])]  ###1.因子的风险暴露*因子的权重 2.约束：添加单位1是为了保证权重和为1  b_eq = A_eq*w 
        bounds = tuple([(0, num_multi * wt_in_index) for wt_in_index in cur_benchmark_wt.values])  #上下界
    else:
        A_ub = A_sector.loc[:,com_code].values ##不等式约束
        b_ub = b_sector
        A_factor = limit_factors.T.values
        target = np.dot(A_factor,cur_benchmark_wt)
        one_vec = np.ones((1,len(limit_factors)))
        
        total_factor = 0.25
        total_sum = 0.1
        
        A_ub_2 = np.vstack([A_factor,-A_factor])
        b_ub_2 = np.hstack([target + total_factor,-target + total_factor])
        
        A_ub_2 = np.vstack([A_ub_2,one_vec,-one_vec])
        b_ub_2 = np.hstack([b_ub_2,1 + total_sum,-1 + total_sum])

        A_ub = np.vstack([A_ub,A_ub_2])
        b_ub = np.hstack([b_ub,b_ub_2])
        A_eq = None
        b_eq = None     
        bounds = tuple([(0, num_multi * wt_in_index) for wt_in_index in cur_benchmark_wt.values])  #上下界
    res = linprog(-c, A_ub, b_ub, A_eq, b_eq, bounds) ##求解最优权重
    cur_wt = pd.Series(res.x, index=cur_est_rets.index)

    return cur_wt

def linear_programming(data_dict):
    """
    线性规划法-求解最优组合权重
    """
    est_stock_rets, limit_fac_data, index_wt = data_dict['est_stock_rets'], data_dict['limit_fac_data'], data_dict['index_wt']
    stock_wt = pd.DataFrame()
    ret_60 = get_market_status()
    defensive_sectors = [
    '银行', '保险', '多元金融',
    '食品', '啤酒', '软饮料', '乳制品', '白酒', '红黄酒',
    '中成药', '化学制药', '生物制药', '医药商业', '医疗保健',
    '种植业', '农业综合', '渔业', '林业', '饲料',
    '水务', '供气供热',
    '公共交通', '公路', '铁路', '机场', '港口', '仓储物流',
    '电信运营',
    '环境保护'
]
    all_industry = get_factor(['industry'])['industry']
    for date in est_stock_rets.columns:
        est_rets = est_stock_rets[date]#date+1期的股票预期收益率
        date_datetime = pd.to_datetime(date)
        limit_fac_panel = limit_fac_data[date_datetime]  #date期的风险因子
        benchmark_wt = index_wt[date].dropna()  #date期的基准指数成分权重
        est_rets, limit_fac_panel = est_rets.loc[benchmark_wt.index], limit_fac_panel.loc[benchmark_wt.index]
        if ret_60.loc[date] == "bad":
            sector_upper = {}
            print("经济下行")
            date_month_end = pd.to_datetime(date) + MonthEnd(0)
            #key = date.strftime("%Y-%m-%d")
            for ind in industry_list:
                is_def = any(k in str(ind) for k in defensive_sectors)
                sector_upper[ind] = 0.3 if is_def else 0.15         
            ind_series = all_industry.loc[benchmark_wt.index, date]
            ind_dummy = pd.get_dummies(ind_series, prefix="sector").astype(int)
            ind_dummy = ind_dummy.reindex(columns=[s for s in industry_list], fill_value=0)
            A_sector = ind_dummy.T
            b_sector = np.array([sector_upper[s] for s in industry_list])*1.0
        else:
            A_sector = None
            b_sector = None
        #求解date期股票权重向量,将来在date+1期股票开盘的时候,根据这个权重向量进行股票买卖,开仓,调仓等
        cur_wt = lp_solve(est_rets, limit_fac_panel, benchmark_wt,A_sector,b_sector)
        cur_wt.name = date
        stock_wt = pd.concat([stock_wt, cur_wt], axis=1)
    stock_wt = stock_wt.where(stock_wt != 0, np.nan)
    return stock_wt

def performance_attribution(factors_dict, index_wt, stock_wt, est_fac_rets, start_date, end_date):
    """
    业绩归因
    """
    factors_panel = concat_factors_panel(None, factors_dict, None, False, False)
    dates = stock_wt.loc[:, start_date:end_date].columns
    dates = pd.to_datetime(dates)
    res = pd.DataFrame()
    for date in dates:
        date_str = date.strftime("%Y%m%d")
        cur_index_wt = index_wt[date_str] / 100
        cur_index_wt = cur_index_wt / cur_index_wt.sum()
        w_delta = stock_wt[date_str] - cur_index_wt  ##最有权重 - 基准权重
        w_delta = w_delta.dropna()
        try:
            cur_factors_panel = factors_panel[date].loc[w_delta.index, :]
        except:
            print(factors_panel[date])
            print(w_delta)
            raise 
        cur_factor_exposure = w_delta.T @ cur_factors_panel  #权重变化*因子暴露
        cur_factor_exposure.name = date
        res = pd.concat([res, cur_factor_exposure], axis=1)

    res = res.T.groupby(pd.Grouper(freq='YE')).mean()
    return res

def machine_learning_model(data):
    def train_model():
        X_train = panel_x.loc[(panel_x.index.get_level_values('date') < val_year) & (panel_x.index.get_level_values('date') >=start_year)]
        y_train = panel_y.loc[(panel_y.index.get_level_values('date') < val_year) & (panel_y.index.get_level_values('date') >=start_year)]
        X_val = panel_x.loc[(panel_x.index.get_level_values('date') >= val_year) & (panel_x.index.get_level_values('date') <split_year)]
        y_val = panel_y.loc[(panel_y.index.get_level_values('date') >= val_year) & (panel_y.index.get_level_values('date') <split_year)]
        
        data_common = pd.concat([X_train,y_train],axis=1)
        data_common = data_common.dropna(how="any",axis=0)
        X_train = data_common.iloc[:,:-1]
        y_train =  data_common.iloc[:,-1]

        data_common = pd.concat([X_val,y_val],axis=1)
        data_common = data_common.dropna(how="any",axis=0)
        X_val = data_common.iloc[:,:-1]
        y_val =  data_common.iloc[:,-1]
        
        model = CatBoostRegressor(
            iterations=800,
            learning_rate=0.03,
            max_depth=5,
            l2_leaf_reg=3,
            random_strength=0.1,
            random_state=42,
            verbose=100,
            allow_writing_files=False
        )
        # model = lgb.LGBMRegressor(
        #                             n_estimators=800,
        #                             learning_rate=0.05,
        #                             max_depth=3,
        #                             min_child_samples=20,
        #                             subsample=0.7,               # 行采样
        #                             colsample_bytree=0.8,        # 列采样
        #                             verbose=-1)

        #early_stopping = lgb.early_stopping(stopping_rounds=50,verbose=False)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return model
    def test_model(model):
    
        X_test = panel_x.loc[(panel_x.index.get_level_values('date') <= end_year) & (panel_x.index.get_level_values('date') >=split_year)]
        X = X_test.dropna(how="any",axis=0)
        y_test_pre = model.predict(X)
        
        y_test_pre = pd.DataFrame(y_test_pre,index=X.index)

        print(y_test_pre)
        return y_test_pre
    panel_y = get_factor(['PCT_CHG_NM'])['PCT_CHG_NM'].stack(level=0)
    panel_y.index.names = ['code', 'date']  # 命名索引
    panel_y = panel_y.swaplevel() 
    panel_x = data
    est_stock_return = pd.DataFrame()
    for year in range(2014,2020):
        split_year = pd.to_datetime(f"{year}-01-01")
        start_year = pd.to_datetime(f"{year-2}-01-01")
        val_year = pd.to_datetime(f"{year-1}-01-01")
        end_year = pd.to_datetime(f"{year}-12-31")
        est_stock_return = pd.concat([test_model(train_model()),est_stock_return])
    return est_stock_return

def get_market_data(use_pctchg=True):
    """
    提取涨跌幅
    """
    global factor_generater
    if use_pctchg:
        market_data = factor_generater.pct_chg
    else:
        market_data = factor_generater.hfq_close
        market_data = market_data.ffill(axis=1).bfill(axis=1)
    market_data.columns = pd.to_datetime(market_data.columns)
    return market_data

def get_ori_name(factor_name, factors_to_concat):
    ''' 如果因子是合成因子或者是正交过的因子,取它们的原始名称
    '''
    if 'ortho' in factor_name: #先判断是否被正交过
        factor_name = factor_name[:-6]
    if 'con' in factor_name: #再判断是否合成因子
        pat = re.compile('(.*)_con_')  ##定义正则规则
        ori_name = re.findall(pat, factor_name)[0]   #匹配
        return factors_to_concat[ori_name]
    else:
        return [factor_name]

def factor_process(method, factors_to_concat, factors_ortho, index_wt, mut_codes, factors, risk_factors=None):
    #因子合成（等权）
    for factor_con, factors_to_con in factors_to_concat.items():
        factor_concat(factors_to_con, factor_con) #合成后的因子将保存到特定目录
    print("完成因子合成")
    #因子正交
    for factor_x, factors_y in factors_ortho.items():
        orthogonalize(factors_y, factor_x, None, index_wt) #正交后的因子将保存到特定目录
    print("完成因子正交")


def index_enhance_model(method='l', benchmark='000300.SH', start_date=None, end_date=None, methods=None):
    global index_enhance_dir,industry_list
    lp_save_path = os.path.join(index_enhance_dir, '线性规划')
    ss_save_path = os.path.join(index_enhance_dir, '机器学习')

    if method == 'l':
        method_name = 'linear_programming'
        save_path = lp_save_path
    elif method == 'm':
        method_name = 'machine_learing'
        save_path = ss_save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    wt_save_path = os.path.join(save_path, '优化权重')
    if not os.path.exists(wt_save_path):
        os.mkdir(wt_save_path)

    pctchgnm = get_factor(['PCT_CHG_NM'])['PCT_CHG_NM']
    index_wt = get_stock_wt_in_index(benchmark) #基准指数权重
    mut_codes = index_wt.index.intersection(pctchgnm.index)

    data_dict = {} #核心权重优化函数的参数

    params = methods[method_name]  ## methods--是参数集合
    factors, risk_factors, window, half_life = params['factors'], params['risk_factors'], params['window'], params['half_life']

   

    if method == 'l':
        factors_dict = {fac: get_factor([fac], mut_codes)[fac] for fac in factors} #读取阿尔法因子的矩阵数据
        factors_panel = concat_factors_panel(None, factors_dict, mut_codes)
        #利用
        est_fac_rets = factor_return_forecast(factors, factors_panel, window, half_life) #因子收益率预测
        est_fac_rets = est_fac_rets[factors]
        est_stock_rets = get_est_stock_return(factors, factors_panel, est_fac_rets, window, half_life) #计算股票预期收益率
        print('计算股票预期收益率完成...')
    elif method == 'm':
        factors_dict = {fac: get_factor([fac])[fac] for fac in factors} #读取阿尔法因子的矩阵数据
        factors_panel = concat_factors_panel(None, factors_dict,None)
        factors_data = pd.concat(
        [pd.DataFrame(factors_panel[date]) for date in factors_panel.keys()],
        keys=factors_panel.keys(),
        names=['date', 'code']
    )

        factors_data[industry_list] = factors_data[industry_list].fillna(0)
        est_fac_rets = factor_return_forecast(factors, factors_panel, window, half_life) #因子收益率预测
        est_fac_rets = est_fac_rets[factors]
        est_stock_rets = machine_learning_model(factors_data) #计算股票预期收益率
        est_stock_rets = est_stock_rets.unstack(level=0)
        est_stock_rets.columns = est_stock_rets.columns.droplevel(0)
        est_stock_rets.columns.name = None

        print('计算股票预期收益率完成...')

    risk_fac_data = {fac: get_factor([fac], mut_codes)[fac] for fac in ['industry', 'MKT_CAP_FLOAT']}
    data_dict.update(risk_fac_data) 
    
    limit_fac_data = concat_factors_panel(risk_factors, risk_fac_data, mut_codes, ind=True, mktcap=False) #矩阵转换成截面,同时加入行业伪变量  
    data_dict.update({'limit_fac_data': limit_fac_data})
    est_stock_rets.columns = est_stock_rets.columns.strftime("%Y%m%d")
    mut_dates = index_wt.columns.intersection(est_stock_rets.columns)
    mut_codes = mut_codes.intersection(est_stock_rets.index)
    index_wt = index_wt.loc[mut_codes, mut_dates]
    est_stock_rets = est_stock_rets.loc[mut_codes, mut_dates]
    print(est_stock_rets)

    est_stock_rets.name = 'est_stock_return'
    data_dict.update({'index_wt': index_wt, 'est_stock_rets': est_stock_rets})
    #开始优化股票权重
      ##globals类似类，可以调用method_name同名函数
    #重点:输入参数是t+1期(期末)股票预期收益率,输出结果是t期(期末)股票权重或者说是t+1期期初的股票权重
    stock_wt = linear_programming(data_dict)
    stock_wt = stock_wt / stock_wt.sum() #权重归一化
    print('计算股票权重完成...')

    #股票权重分析
    stock_wt.to_csv(os.path.join(wt_save_path, 'stock_wt.csv'), encoding="utf-8")
    # stock_weights_analysis(wt_save_path, stock_wt, index_wt)
    print('股票权重分析完成...')

    #接下来为回测做准备
    all_codes = stock_wt.index
    market_data = get_market_data(use_pctchg=False)
    benchmarkdata = market_data.loc[benchmark, start_date:end_date].T #基准指数日涨跌幅
    market_data = market_data.loc[all_codes, start_date:end_date] #基准指数所有成分股票的日涨跌幅
    market_status = get_market_status()
    #根据优化得到的各月末截面期HS300成分股股票权重,进行回测
    bt = Backtest_stock(market_data=market_data, 
                        start_date=start_date, 
                        end_date=end_date, 
                        benchmarkdata=benchmarkdata, 
                        stock_weights=stock_wt, 
                        use_pctchg=False,
                        market_status=market_status)
    print("======开始回测")
    bt.run_backtest()
    print('回测结束, 进行回测结果分析...')
    summary_yearly = bt.summary_yearly() #回测统计
    summary_yearly.to_csv(os.path.join(save_path, f'回测统计_{start_date}至{end_date}.csv'), encoding="utf-8")
    bt.portfolio_record.to_csv(os.path.join(save_path, f'回测净值_{start_date}至{end_date}.csv'), encoding="utf-8")
    bt.position_record.to_csv(os.path.join(save_path, f'各期持仓_{start_date}至{end_date}.csv'), encoding="utf-8")
    plot_net_value(bt.portfolio_record, benchmark, method_name, save_path, start_date, end_date)

    #业绩归因
    p_attr = performance_attribution(factors_dict, index_wt, stock_wt, est_fac_rets, start_date, end_date)
    p_attr.to_csv(os.path.join(save_path, f'业绩归因_{start_date}至{end_date}.csv'), encoding="utf-8")
    print("分析结果存储完成!")

