# -*- coding: utf-8 -*-
import requests
import json
import pandas as pd
from io import StringIO
import numpy as np
import time

#
timezones={}
#function = 'TIME_SERIES_INTRADAY'
apii = 'https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&outputsize=full&datatype=csv&apikey='
apid = 'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize=full&datatype=csv&apikey='
#https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=ASML&interval=1min&outputsize=compact&datatype=csv&time_period=0&apikey=
sector = 'https://www.alphavantage.co/query?function=SECTOR&datatype=csv&apikey='

s_type = ['close','high','low']#,'open']
ma_types = [0,1,2,3,4,5,6,7,8]
#Moving average type By default, matype=0. INT 0 = SMA, 1 = EMA, 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA).

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
	#125
	out_df = pd.DataFrame()
	first = True

	for t in ma_range:
		for s in s_type:
			indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t,series_type=s))
			time.sleep(11.8)
			fixed = StringIO(indicator.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		if first:
			out_df = pd.read_csv(fixed)
			first = False
		elif first != True:
			indi_df = pd.read_csv(fixed)
			out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def macdext_get(macd,symbol, interval):#,types=False,time_period=False):
	out_df = pd.DataFrame()
	macd_range = [[5,10,3],[10,20,7],[12,26,9],[15,35,11]]
	api = indicator_dict[macd]
	macd_ma = 1
	first=True
	for i in macd_range:
		for s in s_type:
			indicator = requests.get(api.format(symbol=symbol,interval=interval,series_type=s,fastperiod=i[0],slowperiod=i[1],signalperiod=i[2],fastmatype=ma_types[1],slowmatype=ma_types[1],signalmatype=ma_types[1]))
			time.sleep(11.8)
			fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
			if first:
				out_df = pd.read_csv(fixed)
				first = False
			elif first != True:
				indi_df = pd.read_csv(fixed)
				out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def stoch_get(stoch,symbol,interval):
	slowd = 3
	slowk = 3
	fastk = 5
	fastd = 3
	stoch_ma = 1
	#EMA
	api = indicator_dict[stoch]

	if stoch == 'stoch':
		indicator = requests.get(api.format(symbol=symbol,interval=interval,fastkperiod=fastk,slowkperiod=slowk,slowdperiod=slowd,slowkmatype=stoch_ma,slowdmatype=stoch_ma))
		time.sleep(11.8)
		fixed = StringIO(indicator.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		indi_df = pd.read_csv(fixed)
		return indi_df

	elif stoch == 'stochf':
		indicator = requests.get(api.format(symbol=symbol,interval=interval,fastkperiod=fastk,fastdperiod=fastd,fastdmatype=stoch_ma))
		time.sleep(11.8)
		fixed = StringIO(indicator.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		indi_df = pd.read_csv(fixed)
		return indi_df

def rsi_get(rsi,symbol,interval):
	out_df = pd.DataFrame()
	rsi_period = [7,11,14,21]
	api = indicator_dict[rsi]
	first = True
	for t in rsi_period:
		for s in s_type:
			indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t,series_type=s))
			time.sleep(11.8)
			fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
			if first:
				out_df = pd.read_csv(fixed)
				first = False
			elif first != True:
				indi_df = pd.read_csv(fixed)
				out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def stochrsi_get (indicator,symbol,interval):
	api = indicator_dict[indicator]
	fastk = 5
	fastd = 3
	fastma = 1
	stype = 'close'
	rsi_period = [7,11,14,21]
	first = True
	for t in rsi_period:
		for s in s_type:
			indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t,series_type=s,fastkperiod=fastk,fastdperiod=fastd,fastdmatype=fastma))
			time.sleep(11.8)
			fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
			if first:
				out_df = pd.read_csv(fixed)
				first = False
			elif first != True:
				indi_df = pd.read_csv(fixed)
				out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def adx_get(indicator,symbol,interval):
	api = indicator_dict[indicator]
	adx_period = [7,11,14,21]
	first = True
	for t in adx_period:
		indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t))
		time.sleep(11.8)
		fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		if first:
			out_df = pd.read_csv(fixed)
			first = False
		elif first != True:
			indi_df = pd.read_csv(fixed)
			out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def cci_get(indicator,symbol,interval):
	api = indicator_dict[indicator]
	cci_range = [5,10,15,20,35,50,65,85,100,125,200,250]
	first = True
	#annual/time period cycle high to high divided by three is the official time period
	for t in cci_range:
		indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t))
		time.sleep(11.8)
		fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		if first:
			out_df = pd.read_csv(fixed)
			first = False
		elif first != True:
			indi_df = pd.read_csv(fixed)
			out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def aroon_get(indicator,symbol,interval):
	api= indicator_dict[indicator]
	aroon_range = [5,10,15,20,35,50,65,85,100,125,200,250]
	#period since last highest high and lowest low
	first = True
	for t in aroon_range:
		indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t))
		time.sleep(11.8)
		fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		if first:
			out_df = pd.read_csv(fixed)
			first = False
		elif first != True:
			indi_df = pd.read_csv(fixed)
			out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def bbands_get(indicator,symbol,interval):
	api= indicator_dict[indicator]
	bb_range = [5,10,15,20,35,50,65,100,125,200,250]
	ndup = 2
	nddn = 2
	bband_ma = [0,1,4]
	first = True
	for t in bb_range:
		for m in bband_ma:
			for s in s_type:
				indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t,series_type=s,nbdevup=ndup,nbdevdn=nddn,matype=m))
				time.sleep(11.8)
				fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
				if first:
					out_df = pd.read_csv(fixed)
					first = False
				elif first != True:
					indi_df = pd.read_csv(fixed)
					out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def adosc_get(indicator,symbol,interval):
	api= indicator_dict[indicator]
	fastperiod = 3
	slowperiod = 10
	first = True
	#2,7?
	indicator = requests.get(api.format(symbol=symbol,interval=interval,fastperiod=fastperiod,slowperiod=slowperiod))
	time.sleep(11.8)
	fixed = StringIO(indicator.content.decode('utf-8'))
#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
	out_df = pd.read_csv(fixed)
	return out_df


def simple_indicator(indicator,symbol,interval):
	api = indicator_dict[indicator]
	indicator = requests.get(api.format(symbol=symbol,interval=interval))
	time.sleep(11.8)
	fixed = StringIO(indicator.content.decode('utf-8'))
#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
	indi_df = pd.read_csv(fixed)
	return indi_df

indicator_run = {
	'sma':moving_a,
	'ema':moving_a,
	'tema':moving_a,
	'macd':simple_indicator,
	'macdext':macdext_get,
	'stoch':stoch_get,
	'stochf':stoch_get,
	'rsi':rsi_get,
	'stochrsi':stochrsi_get,
	'willr':'notassigned',
	'adx':adx_get,
	'adxr':adx_get,
	'apo':'notassigned',
	'ppo':'notassigned',
	'mom':'notassigned',
	'bop':simple_indicator,
	'cci':cci_get,
	'cmo':'notassigned',
	'roc':'notassigned',
	'rocr':'notassigned',
	'aroon':aroon_get,
	'aroonosc':aroon_get,
	'mfi':'notassigned',
	'trix':'notassigned',
	'ultosc':'notassigned',
	'dx':'notassigned',
	'minus_di':'notassigned',
	'plus_di':'notassigned',
	'minus_dm':'notassigned',
	'plus_dm':'notassigned',
	'bbands':bbands_get,
	'midpoint':'notassigned',
	'midprice':'notassigned',
	'sar':'notassigned',
	'trange':simple_indicator,
	'atr':'notassigned',
	'natr':'notassigned',
	'ad':simple_indicator,
	'adosc':adosc_get,
	'obv':simple_indicator,
	'ht_trendline':'notassigned',
	'ht_sine':'notassigned',
	'ht_trendmode':'notassigned',
	'ht_dcperiod':'notassigned',
	'ht_dcphase':'notassigned',
	'ht_dcphasor':'notassigned'
	}

def get_data (symbol,interval):
	if interval in ['1min','5min','15min','30min','60min']:
		intra_csv = requests.get(apii.format(function = 'TIME_SERIES_INTRADAY',symbol=symbol,interval=interval))
		time.sleep(11.8)
		#response 200 = got it
		fixed = StringIO(intra_csv.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		intra_df = pd.read_csv(fixed)
		indicator_df = get_indicators(symbol,interval)
		out_df = pd.merge(intra_df,indicator_df,on='time',how='inner')
		return out_df
	elif interval == 'daily':
		daily_csv = requests.get(apid.format(function = 'TIME_SERIES_DAILY',symbol=symbol))
		time.sleep(11.8)
		#response 200 = got it
		indicator_df = get_indicators(symbol,interval)
		fixed = StringIO(daily_csv.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		d_df = pd.read_csv(fixed)

		print(d_df)
		print(indicator_df)
		out_df = pd.merge(d_df,indicator_df,left_index=True,right_index=True,how='inner')
		return out_df

def run_live (symbol,interval):
	df = get_data(symbol,interval)
	timer = time.asctime()
	while timer[11:15] != '16:0':
		intra_json = requests.get(api.format(symbol=symbol,interval=interval))
		time.sleep(11.8)
		fixed = StringIO(intra_json.content.decode('utf-8'))
		intra_row_df = pd.read_csv(fixed)
		df.append(intra_row_df,ignore_index = True)
#		time.sleep(60)

def get_indicators (symbol,interval
#df,
):
	indicators = indicator_dict.keys()
	first = True
	for i in indicator_list:
		indi_out = indicator_run[i](i,symbol,interval)
		if first:
			indicator_out = indi_out
			first = False
		elif first != True:
			try:
				indicator_out = pd.merge(indicator_out,indi_out,on='time',how='inner')
			except Exception:
				print(i)
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
'bop',
'cci',
'aroon',
'aroonosc',
'bbands',
'ad',
'adosc',
'obv'
]

if __name__ == '__main__':
	print ('Intervals(#,#,#...)=\n1: 1min\n2: 5min\n3: 15min\n4: 30min\n5: 60min\n6: daily')
	interval = str(6)#int(input('interval')))
	interval = interval.split(',')
	#list of intervals to run per symbol
	interval_dic = {'1':'1min','2':'5min','3':"15min",'4':'30min','5':'60min','6':'daily'}
	
	for symbol in symbols:
		for i in range(len(interval)):
			csv_df = get_data (symbol,interval_dic[interval[i]])
			name = '_'.join([symbol,interval_dic[interval[i]]])
			csv_df.to_csv(name,sep='\t')


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