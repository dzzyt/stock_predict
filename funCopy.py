# -*- coding: utf-8 -*-
import requests
import json
import pandas as pd
from io import StringIO
import numpy as np
import time


#
timezones={}
function = 'TIME_SERIES_INTRADAY'
api = 'https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&outputsize=full&datatype=csv&apikey='
apid = 'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize=full&datatype=csv&apikey='
#https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=ASML&interval=1min&outputsize=compact&datatype=csv&time_period=0&apikey=
sector = 'https://www.alphavantage.co/query?function=SECTOR&datatype=csv&apikey='

s_type = ['open','close','high','low']
ma_types = [1,2,3,4,5,6,7,8]
#Moving average type By default, matype=0. INT 0 = SMA, 1 = EMA, 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA).


indicator_run = {
	'sma':moving_a,
	'ema':moving_a,
	'tema':moving_a,
	'macd':simple_indicator,
	'macdext':macdext_get,
	'stoch',
	'stochf',
	'rsi',
	'stochrsi',
	'willr',
	'adx',
	'adxr',
	'apo',
	'ppo',
	'mom',
	'bop':simple_indicator,
	'cci',
	'cmo',
	'roc',
	'rocr',
	'aroon',
	'aroonosc',
	'mfi',
	'trix',
	'ultosc',
	'dx',
	'minus_di',
	'plus_di',
	'minus_dm',
	'plus_dm',
	'bbands',
	'midpoint',
	'midprice',
	'sar',
	'trange':simple_indicator,
	'atr',
	'natr',
	'ad':simple_indicator,
	'adosc',
	'obv':simple_indicator,
	'ht_trendline',
	'ht_sine',
	'ht_trendmode',
	'ht_dcperiod',
	'ht_dcphase',
	'ht_dcphasor'
	}

params = ['function',
'symbol',
'interval',
'additional_arguments',
'time_period',
'series_type',
'fastperiod',
'slowperiod',
'signalperiod',
'fastmatype',
'slowmatype',
'signalmatype',
'fastkperiod',
'slowkperiod',
'slowdperiod',
'slowkmatype',
'slowdmatype',
'fastdperiod',
'fastdmatype',
'timeperiod1',
'timeperiod2',
'timeperiod3',
'nbdevup',
'nbdevdn'
]

indicator_dict = {
	'sma':'https://www.alphavantage.co/query?function=SMA&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=',
	'ema':'https://www.alphavantage.co/query?function=EMA&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=',
	'tema':'https://www.alphavantage.co/query?function=TEMA&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=',
	'macd':'https://www.alphavantage.co/query?function=MACD&symbol={symbol}&interval={interval}&series_type=close&fastperiod=12&slowperiod=26&signalperiod=9&datatype=csv&apikey=',
	'macdext':'https://www.alphavantage.co/query?function=MACDEXT&symbol={symbol}&interval={interval}&series_type={series_type}&fastperiod={fastperiod}&slowperiod={slowperiod}&signalperiod={signalperiod}&fastmatype={fastmatype}&slowmatype={slowmatype}&signalmatype={signalmatype}&datatype=csv&apikey=',
	'stoch':'https://www.alphavantage.co/query?function=STOCH&symbol={symbol}&interval={interval}&fastkperiod={fastkperiod}&slowkperiod={slowkperiod}&slowdperiod={slowdperiod}&slowkmatype={slowkmatype}&slowdmatype={slowdmatype}&datatype=csv&apikey=',
	'stochf':'https://www.alphavantage.co/query?function=STOCHF&symbol={symbol}&interval={interval}&fastkperiod={fastkperiod}&fastdperiod={fastdperiod}&fastdmatype={fastdmatype}&datatype=csv&apikey=',
	'rsi':'https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=',
	'stochrsi':'https://www.alphavantage.co/query?function=STOCHRSI&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&fastkperiod={fastkperiod}&fastdperiod={fastdperiod}&fastdmatype={fastdmatype}&datatype=csv&apikey=',
	'willr':'https://www.alphavantage.co/query?function=WILLR&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'adx':'https://www.alphavantage.co/query?function=ADX&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'adxr':'https://www.alphavantage.co/query?function=ADXR&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'apo':'https://www.alphavantage.co/query?function=APO&symbol={symbol}&interval={interval}&series_type={series_type}&fastperiod={fastperiod}&slowperiod={slowperiod}&matype={matype}&datatype=csv&apikey=',
	'ppo':'https://www.alphavantage.co/query?function=PPO&symbol={symbol}&interval={interval}&series_type={series_type}&fastperiod={fastperiod}&slowperiod={slowperiod}&matype={matype}&datatype=csv&apikey=',
	'mom':'https://www.alphavantage.co/query?function=MOM&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=',
	'bop':'https://www.alphavantage.co/query?function=BOP&symbol={symbol}&interval={interval}&datatype=csv&apikey=',
	'cci':'https://www.alphavantage.co/query?function=CCI&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'cmo':'https://www.alphavantage.co/query?function=CMO&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=',
	'roc':'https://www.alphavantage.co/query?function=ROC&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=',
	'rocr':'https://www.alphavantage.co/query?function=ROCR&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=',
	'aroon':'https://www.alphavantage.co/query?function=AROON&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'aroonosc':'https://www.alphavantage.co/query?function=AROONOSC&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'mfi':'https://www.alphavantage.co/query?function=MFI&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'trix':'https://www.alphavantage.co/query?function=TRIX&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=',
	'ultosc':'https://www.alphavantage.co/query?function=ULTOSC&symbol={symbol}&interval={interval}&timeperiod1={timeperiod1}&timeperiod2={timeperiod2}&timeperiod3={timeperiod3}&datatype=csv&apikey=',
	'dx':'https://www.alphavantage.co/query?function=DX&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'minus_di':'https://www.alphavantage.co/query?function=MINUS_DI&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'plus_di':'https://www.alphavantage.co/query?function=PLUS_DI&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'minus_dm':'https://www.alphavantage.co/query?function=MINUS_DM&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'plus_dm':'https://www.alphavantage.co/query?function=PLUS_DM&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'bbands':'https://www.alphavantage.co/query?function=BBANDS&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&nbdevup={nbdevup}&nbdevdn={nbdevdn}&matype={matype}&datatype=csv&apikey=',
	'midpoint':'https://www.alphavantage.co/query?function=MIDPOINT&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=',
	'midprice':'https://www.alphavantage.co/query?function=MIDPRICE&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'sar':'https://www.alphavantage.co/query?function=SAR&symbol={symbol}&interval={interval}&acceleration={acceleration}&maximum={maximum}&datatype=csv&apikey=',
	'trange':'https://www.alphavantage.co/query?function=TRANGE&symbol={symbol}&interval={interval}&datatype=csv&apikey=',
	'atr':'https://www.alphavantage.co/query?function=ATR&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'natr':'https://www.alphavantage.co/query?function=NATR&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=',
	'ad':'https://www.alphavantage.co/query?function=AD&symbol={symbol}&interval={interval}&datatype=csv&apikey=',
	'adosc':'https://www.alphavantage.co/query?function=ADOSC&symbol={symbol}&interval={interval}&fastperiod={fastperiod}&slowperiod={slowperiod}&datatype=csv&apikey=',
	'obv':'https://www.alphavantage.co/query?function=OBV&symbol={symbol}&interval={interval}&datatype=csv&apikey=',
	'ht_trendline':'https://www.alphavantage.co/query?function=HT_TRENDLINE&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=',
	'ht_sine':'https://www.alphavantage.co/query?function=HI_SINE&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=',
	'ht_trendmode':'https://www.alphavantage.co/query?function=HT_TRENDMODE&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=',
	'ht_dcperiod':'https://www.alphavantage.co/query?function=HT_DCPERIOD&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=',
	'ht_dcphase':'https://www.alphavantage.co/query?function=HT_DCPHASE&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=',
	'ht_dcphasor':'https://www.alphavantage.co/query?function=HT_DCPHASOR&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey='
}

def moving_a(ma,symbol,interval):
	api = indicator_dict[ma]
	ma_range = [5,10,15,20,35,50,65,100,125,200,250]
	out_df = pd.DataFrame()

	for t in ma_range:
		for s in series_type:
			indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t,series_type=s))
			fixed = StringIO(indicator.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
			indi_df = pd.read_csv(fixed)
			out_df = pd.merge(out_df,indi_df,left_index=True,right_index=True,how="outer")
			#might want to use date time column run and see what the column is called

def macdext_get(macd,symbol, interval):#,types=False,time_period=False):
	macd_range = [[5,10,3],[10,20,7],[12,26,9],[15,35,11]]
	api = indicator_dict[macd]
	for i in macd_range:


def stoch_get(stoch,symbol,interval):
	slowd = 3
	slowk = 3
	fastk = 5
	fastd = 3
	stoch_ma = 1
	#EMA
	api = indicator_dict[stoch]


def rsi_get(rsi,symbol,interval):
	rsi_period = [7,11,14,21]
	api = indicator_dict[rsi]
	for i in rsi_period:


def simple_indicator(indicator,symbol,interval):
	api = indicator_dict[indicator]
	indicator = requests.get(api.format(symbol=symbol,interval=interval))
	fixed = StringIO(indicator.content.decode('utf-8'))
#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
	indi_df = pd.read_csv(fixed)

	return indi_df



def merge_df():
	


def get_data (symbol,interval):
	if interval in ['1min','5min','15min','30min','60min']:

		intra_csv = requests.get(api.format(function = 'TIME_SERIES_INTRADAY',symbol=symbol,interval=interval))
		#response 200 = got it
		fixed = StringIO(intra_csv.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		intra_df = pd.read_csv(fixed)
		indicator_df = get_indicators()
		out_df = merge_df()
		return out_df
	elif interval == 'daily':
		daily_csv = requests.get(apid.format(function = 'TIME_SERIES_DAILY',symbol=symbol))
		#response 200 = got it
		fixed = StringIO(daily_csv.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		d_df = pd.read_csv(fixed)
		indicator_df = get_indicators()
		out_df = merge_df()
		return out_df

def run_live (symbol,interval):
	df = get_data(symbol,interval)
	timer = time.asctime()

	while timer[11:15] != '16:0':

		intra_json = requests.get(api.format(symbol=symbol,interval=interval))
		fixed = StringIO(intra_json.content.decode('utf-8'))
		intra_row_df = pd.read_csv(fixed)
		df.append(intra_row_df,ignore_index = True)
#		time.sleep(60)



def get_indicators (symbol,interval
#df,
):

	indicators = indicator_dict.keys()
	indicator_out = pd.DataFrame()

	for i in indicator_list:
			indi_out = indicator_run[i](i,symbol,interval)
			indicator_out = merge_df()

	return indicator_out

	#aroon lagging does not predict change, asesses trends strength

#	df = df.assign(
#	row_name = Some_series e.g.(=some_list, =lambda x(row): x['colum_name'] + x['another_column'] * some_math_function)
#		)

symbols = ['ASML']
indicator_list = [
'sma',
'ema',
'tema',
'macd',
'macdext',
'stoch',
'stochf',
'rsi',
'stochrsi',
'adx',
'adxr',
'cci',
'aroon',
'aroonosc',
'bbands',
'ad',
'adosc',
'obv'
]

if __name__ == '__main__':
	print ('Intervals(#,#,#...)=\n1: 1min\n2: 5min\n3: 15min\n4: 30min\n5: 60min')
	interval = str(6)#int(input('interval')))
	interval = interval.split(',')
	#list of intervals to run per symbol
	interval_dic = {'1':'1min','2':'5min','3':"15min",'4':'30min','5':'60min','6':'daily'}
	
	for symbol in symbols:
		for i in range(len(interval)):
			get_data (symbol,interval_dic[interval[i]])
