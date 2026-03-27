import os
import warnings
warnings.filterwarnings('ignore')  #将运行中的警告信息设置为“忽略”，从而不在控制台显示
from index_enhance import *

#工作目录，存放代码
work_dir = os.path.dirname(os.path.dirname(__file__))

def main(method):
    if method == 'm': #机器学习
        method_name = 'machine_learing'
    elif method == 'l': #多因子模型
        method_name = 'linear_programming'
    #要进行大类因子合成的因子
    factors_to_concat = {
            'mom': ['exp_wgt_return_1m', 'exp_wgt_return_3m', 'exp_wgt_return_6m', 'exp_wgt_return_12m'],
            'liq_barra': ['STOA_barra', 'STOM_barra', 'STOQ_barra'],
            'vol': ['std_1m', 'std_3m', 'std_6m', 'std_12m'],
            'growth': ["ROA_ttm","OCF_G_q","assetturnover_ttm"],
            #'lev': ['BLEV_barra', 'DTOA_barra', 'MLEV_barra'],
            }
    #要进行正交的因子
    factors_ortho = {
        
            'liq_barra_con_equal': ['mom_con_equal'],
            'mom_con_equal': ["liq_barra_con_equal"],
            "LNCAP_barra":['LNCAP_barra',"ROA_ttm","RSTR_barra","std_1m","turn_1m","bias_turn_1m"]
            #"vol_con_equal":["liq_barra_con_equal_ortho",'mom_con_equal_ortho']
            #'ROE_q': ['EP'],
            }
    methods = {
          'linear_programming': 
                {'factors': ['mom_con_equal_ortho', 'liq_barra_con_equal_ortho',"growth_con_equal"], #阿尔法因子
                 'risk_factors': ['LNCAP_barra',"SP"], #风险因子
                 'window': 24,
                 'half_life': 1},
            'machine_learing': { 
                'factors': ["ROA_ttm_ortho","RSTR_barra_ortho","std_1m_ortho","turn_1m_ortho","bias_turn_1m_ortho"], #阿尔法因子
                'risk_factors': ["LNCAP_barra"], #风险因子
                'window': 24,
                'half_life': 1
},}


    start_date = '2014-01-30'
    end_date = '2019-12-31'
    benchmark = '000300.SH'
    factors = methods[method_name]['factors']
    risk_factors = methods[method_name]['risk_factors']
    print('开始运行模型...')
    print('*'*80)
    pctchgnm = get_factor(['PCT_CHG_NM'])['PCT_CHG_NM']
    print("数据获取成")
    index_wt = get_stock_wt_in_index(benchmark)
    mut_codes = index_wt.index.intersection(pctchgnm.index)
    print('开始进行因子合成与正交处理...')
    factor_process(method, factors_to_concat, factors_ortho, index_wt, mut_codes, factors, risk_factors)
    print('因子处理完成！')
    print('开始运行指数增强模型...')
    index_enhance_model(method, benchmark, start_date, end_date, methods)

if __name__ == '__main__':
    method = input("请选择指数增强模型方法（'l'-多因子模型; 'm'-机器学习）: ")
    if method not in ('l', 'm'):
        raise TypeError(f"暂不支持的方法：{method}, 请重新运行并输入")
    main(method)