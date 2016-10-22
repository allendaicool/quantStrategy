from sqlalchemy import desc
import numpy as np
import pandas as pd
from dateutil import relativedelta
from sklearn import preprocessing
from sets import Set

 


def initialize(context):
    run_daily(daily_select, time='before_open')
    context.last_update_dt = None
    g.N = 5
    g.weight = 1/(float)(g.N)
    set_slip_fee(context)
    g.max_Num_candidates = 5;
    # g.last_high ={}
     # 止盈百分比
    g.cut_gain_percentage = 0.05
    # 止损百分比
    g.cut_loss_percentage = 0.05
    
    g.in_trend_days = 5
    
    g.ma_lengths = [5,10]
    
    g.stock_stop_range = 0.02
    g.stock_stop_days_period = 2
    g.break_stock_stop_chg_rate = 0.03


    
    
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(PriceRelatedSlippage(0.002))
    # 根据不同的时间段设置手续费
    dt=context.current_dt
    log.info(type(context.current_dt))
    
    if dt>datetime.datetime(2013,1, 1):
        set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5)) 
        
    elif dt>datetime.datetime(2011,1, 1):
        set_commission(PerTrade(buy_cost=0.001, sell_cost=0.002, min_cost=5))
            
    elif dt>datetime.datetime(2009,1, 1):
        set_commission(PerTrade(buy_cost=0.002, sell_cost=0.003, min_cost=5))
                
    else:
        set_commission(PerTrade(buy_cost=0.003, sell_cost=0.004, min_cost=5))

def check_date(context):
    if context.last_update_dt is None: return True
    delta = relativedelta.relativedelta(weeks=1)
    dt = context.last_update_dt + delta
    if context.current_dt.year == dt.year and context.current_dt.month == dt.month: return True
    else: return False
    


def unpaused_not_ST_stock_non_high_limit(stockspool, context):
    current_data=get_current_data()
    return [s for s in stockspool if not current_data[s].paused and not current_data[s].is_st] #and not current_data[s].day_open == current_data[s].high_limit
    
#去除st 股票
def remove_risky_stocks(stocks):
    stock_name_series = stocks['display_name']
    stock_name_list = list(stock_name_series)
    star_stock_index = [stock_name_list.index(s) for s in 
        stock_name_list if s.startswith('*')]
        
    stocks = stocks.drop(stocks.index[star_stock_index])
    # stock_name_series_2 = stocks['display_name']
    # stock_name_series_2[stock_name_series_2.isin(['*欣泰'])]
    
    stocks = stocks.index
    return stocks


def get_stocks(context):
    stocks = get_all_securities(['stock'])
   
    #stocks = stocks.index
    stocks = remove_risky_stocks(stocks)

    unpaused_stock_list = unpaused_not_ST_stock_non_high_limit(stocks, context)
    return unpaused_stock_list
  
def exclude_reach_high_limit_stock(all_past_prices_df_close, all_past_prices_df_open, all_past_prices_df_high_limit, all_past_prices_df_low, all_past_prices_df_high):
    close_price = all_past_prices_df_close.iloc[-1]
    open_price = all_past_prices_df_open.iloc[-1]
    high_limit = all_past_prices_df_high_limit.iloc[-1]
    low_price = all_past_prices_df_low.iloc[-1]
    high_price = all_past_prices_df_high.iloc[-1]
    
    close_open_index = close_price[close_price== open_price].index.tolist()
    open_high_limit_index = open_price[open_price == high_limit].index.tolist()
    high_limit_low_index = high_limit[high_limit == low_price].index.tolist()
    low_high_index = low_price[low_price==high_price].index.tolist()
    unwant_stock = list(set(close_open_index).intersection(open_high_limit_index).intersection(high_limit_low_index).intersection(low_high_index))
    return unwant_stock
    
    
    
def assign_order(s):
    s.dropna(inplace=True)
    arg_sort = s.argsort()
    s[arg_sort] = np.arange(len(arg_sort))
    
    
def normalize_data(data_set):
    data_set.dropna(inplace=True)
    data_set_list = list(data_set)
    index = data_set.index
    data_normalized = preprocessing.normalize(data_set_list, norm='l2')
    data_series_after_normalization = pd.Series(data_normalized.tolist()[0],
        index)
    return data_series_after_normalization
   
def standardization_data(data_set):
    data_set_not_null = data_set.dropna()
    not_null_index = data_set_not_null.index
    include_null_index = data_set.index
    data_np_array = np.array(data_set_not_null)
    X_scaled = preprocessing.scale(data_np_array)
    data_series_after_standardization =  pd.Series(X_scaled.tolist(),data_set_not_null.index)
    if data_series_after_standardization.isnull().values.any():
        print 'contains null ----------------'
    return data_series_after_standardization
    

def market_risk_drop_stocks():
    index = '000001.XSHG'
    h = attribute_history(index, 5, '1d', ('open','close')) # 取得沪深300过去10天的每天的开盘价, 收盘价
    ma51=h['close'][:2].mean()
    ma52=h['close'][-2:].mean()
    maprecent=(ma52-ma51)/ma51
    sharp_drop_rate = (h['close'][-1] - h['close'][-2])/h['close'][-2]
    # risk_avoid_stock_list.append('600519.XSHG')
        # risk_avoid_stock_list.append('601857.XSHG')
        # risk_avoid_stock_list.append('600028.XSHG')
        # risk_avoid_stock_list.append('601398.XSHG')
    #q = query(valuation).filter(valuation.code.in_(risk_avoid_stock_list))

    if maprecent<-0.05:
        return True;

def daily_select(context):
    
    stocks = get_stocks(context)
    # two_types_stock = get_stocks(context)
    # stocks = two_types_stock["unpaused_stock_list"]
    # prioritized_stocks = two_types_stock["top_priority_stock"]
    # index = '000001.XSHG'
    q = query(valuation, balance).filter(valuation.code.in_(stocks), valuation.pe_ratio < 400, valuation.pe_ratio > 0, balance.total_owner_equities/balance.total_sheet_owner_equities > 0.6)  
   
    if market_risk_drop_stocks():
        context.selected_stocks = []
        context.last_update_dt = context.current_dt
        context.need_update = True
    else:
        if not check_date(context): return
        context.last_update_dt = context.current_dt
        context.need_update = True
        df = get_fundamentals(q)
        s_cmc = pd.Series(df['market_cap'].values, index=df['code'])
       
       
        s_cmc_normalize  = standardization_data(s_cmc)
        
        s_pb = pd.Series(df['pb_ratio'].values, index=df['code'])
        
        s_pb_normalize = standardization_data(s_pb)
        #stocks_name = df['code'].tolist()
        df = history(2, unit='120d', field='close', security_list=df['code'].values.tolist())
        s_gain = (df.iloc[1] - df.iloc[0]) / df.iloc[0]
        s_gain_length_before = len(s_gain)
        s_gain.dropna(inplace=True)
        s_gain_length_after = len(s_gain)
        if s_gain_length_before != s_gain_length_after:
            print ('two length are different ')
        

        #s_gain = []
        # s = 1 * s_cmc_normalize + 0* s_pb_normalize
        if s_gain.empty:
            s = 1 * s_cmc_normalize + 0* s_pb_normalize

        else:
            s_gain_normalize = standardization_data(s_gain)
            #s_tmp = 1 * s_cmc_normalize + 0* s_pb_normalize 
            s = 1 * s_cmc_normalize + 0* s_pb_normalize + 0* s_gain_normalize 
        s = s.dropna()
        s.sort()
        
        
        context.selected_stocks = s.index[:g.N]
        for i in range (20):
            print ('stock is ', s.index[i])


    
def update_position(context,data):
    # current_stocks_num = len(context.portfolio.positions)
    # discard_stock = Set()
    #sell_list = sell_signal(context)
    for p in context.portfolio.positions:
        if p not in context.selected_stocks:
            o = order_target(p, 0)
            # discard_stock.add(p)
            # current_stocks_num -= 1
            # del g.last_high[p]
    for s in context.selected_stocks:
        # if  current_stocks_num < g.N and s not in discard_stock:
        target_value = g.weight * context.portfolio.portfolio_value
        o = order_target_value(s, target_value)

# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context, data):
    if context.need_update is True:
        update_position(context,data)
        context.need_update = False
