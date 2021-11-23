# -*- coding: utf-8 -*-
import requests
import pandas as pd
from io import StringIO
import numpy as np
import time
from functools import reduce
#2SXOFPK5YGV8VIVI
timezones={}
#function = 'TIME_SERIES_INTRADAY'
apii = 'https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&outputsize=full&datatype=csv&apikey=2SXOFPK5YGV8VIVI'
apid = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&datatype=csv&apikey=2SXOFPK5YGV8VIVI'
#https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=ASML&interval=1min&outputsize=compact&datatype=csv&time_period=0&apikey=2SXOFPK5YGV8VIVI
sector = 'https://www.alphavantage.co/query?function=SECTOR&datatype=csv&apikey=2SXOFPK5YGV8VIVI'

s_type = ['close','high','low']#,'open']
ma_types = [0,1,2,3,4,5,6,7,8]
#Moving average type By default, matype=0. INT 0 = SMA, 1 = EMA, 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA).

indicator_dict = {
	'sma':'https://www.alphavantage.co/query?function=SMA&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'ema':'https://www.alphavantage.co/query?function=EMA&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'tema':'https://www.alphavantage.co/query?function=TEMA&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'macd':'https://www.alphavantage.co/query?function=MACD&symbol={symbol}&interval={interval}&series_type=close&fastperiod=12&slowperiod=26&signalperiod=9&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'macdext':'https://www.alphavantage.co/query?function=MACDEXT&symbol={symbol}&interval={interval}&series_type={series_type}&fastperiod={fastperiod}&slowperiod={slowperiod}&signalperiod={signalperiod}&fastmatype={fastmatype}&slowmatype={slowmatype}&signalmatype={signalmatype}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'stoch':'https://www.alphavantage.co/query?function=STOCH&symbol={symbol}&interval={interval}&fastkperiod={fastkperiod}&slowkperiod={slowkperiod}&slowdperiod={slowdperiod}&slowkmatype={slowkmatype}&slowdmatype={slowdmatype}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'stochf':'https://www.alphavantage.co/query?function=STOCHF&symbol={symbol}&interval={interval}&fastkperiod={fastkperiod}&fastdperiod={fastdperiod}&fastdmatype={fastdmatype}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'rsi':'https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'stochrsi':'https://www.alphavantage.co/query?function=STOCHRSI&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&fastkperiod={fastkperiod}&fastdperiod={fastdperiod}&fastdmatype={fastdmatype}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'willr':'https://www.alphavantage.co/query?function=WILLR&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'adx':'https://www.alphavantage.co/query?function=ADX&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'adxr':'https://www.alphavantage.co/query?function=ADXR&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'apo':'https://www.alphavantage.co/query?function=APO&symbol={symbol}&interval={interval}&series_type={series_type}&fastperiod={fastperiod}&slowperiod={slowperiod}&matype={matype}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'ppo':'https://www.alphavantage.co/query?function=PPO&symbol={symbol}&interval={interval}&series_type={series_type}&fastperiod={fastperiod}&slowperiod={slowperiod}&matype={matype}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'mom':'https://www.alphavantage.co/query?function=MOM&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'bop':'https://www.alphavantage.co/query?function=BOP&symbol={symbol}&interval={interval}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'cci':'https://www.alphavantage.co/query?function=CCI&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'cmo':'https://www.alphavantage.co/query?function=CMO&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'roc':'https://www.alphavantage.co/query?function=ROC&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'rocr':'https://www.alphavantage.co/query?function=ROCR&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'aroon':'https://www.alphavantage.co/query?function=AROON&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'aroonosc':'https://www.alphavantage.co/query?function=AROONOSC&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'mfi':'https://www.alphavantage.co/query?function=MFI&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'trix':'https://www.alphavantage.co/query?function=TRIX&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'ultosc':'https://www.alphavantage.co/query?function=ULTOSC&symbol={symbol}&interval={interval}&timeperiod1={timeperiod1}&timeperiod2={timeperiod2}&timeperiod3={timeperiod3}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'dx':'https://www.alphavantage.co/query?function=DX&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'minus_di':'https://www.alphavantage.co/query?function=MINUS_DI&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'plus_di':'https://www.alphavantage.co/query?function=PLUS_DI&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'minus_dm':'https://www.alphavantage.co/query?function=MINUS_DM&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'plus_dm':'https://www.alphavantage.co/query?function=PLUS_DM&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'bbands':'https://www.alphavantage.co/query?function=BBANDS&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&nbdevup={nbdevup}&nbdevdn={nbdevdn}&matype={matype}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'midpoint':'https://www.alphavantage.co/query?function=MIDPOINT&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'midprice':'https://www.alphavantage.co/query?function=MIDPRICE&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'sar':'https://www.alphavantage.co/query?function=SAR&symbol={symbol}&interval={interval}&acceleration={acceleration}&maximum={maximum}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'trange':'https://www.alphavantage.co/query?function=TRANGE&symbol={symbol}&interval={interval}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'atr':'https://www.alphavantage.co/query?function=ATR&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'natr':'https://www.alphavantage.co/query?function=NATR&symbol={symbol}&interval={interval}&time_period={time_period}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'ad':'https://www.alphavantage.co/query?function=AD&symbol={symbol}&interval={interval}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'adosc':'https://www.alphavantage.co/query?function=ADOSC&symbol={symbol}&interval={interval}&fastperiod={fastperiod}&slowperiod={slowperiod}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'obv':'https://www.alphavantage.co/query?function=OBV&symbol={symbol}&interval={interval}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'ht_trendline':'https://www.alphavantage.co/query?function=HT_TRENDLINE&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'ht_sine':'https://www.alphavantage.co/query?function=HI_SINE&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'ht_trendmode':'https://www.alphavantage.co/query?function=HT_TRENDMODE&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'ht_dcperiod':'https://www.alphavantage.co/query?function=HT_DCPERIOD&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'ht_dcphase':'https://www.alphavantage.co/query?function=HT_DCPHASE&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI',
	'ht_dcphasor':'https://www.alphavantage.co/query?function=HT_DCPHASOR&symbol={symbol}&interval={interval}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI'
}

def shifter(shifts,shift_df,direction='up'):
#takes in number of shifts and dataframe, returns a list of dataframes with shifts number of dataframes and a each dataframe shifted columnwise stepwise start at 0 to shifts-1
	if shifts >1:
		if direction =='up':
			output = shifter(shifts-1,shift_df,direction)
			output.append(shift_df.shift(0-shifts+1))
			return output
	elif shifts ==1:
		output = [shift_df]
		return output

def moving_a(df,ma,symbol,interval):
#	api = indicator_dict[ma]
	ma_range = [5,10,15,20,35,50,65,100,125,200,250]
	new_series = pd.DataFrame()
	for i in range(len(ma_range)):
		for s in s_type:
			t = ma_range[i]
			name = '_'.join(['sma',str(t),s])
			new_series[name] = reduce(lambda x,y: x.add(y),shifter(t,df[s])).values/t
			name = '_'.join(['ema',str(t),s])
			counter = (len(new_series)-t+1)
			calc_ema = []
			k = 2/(t+1)
			for n in range(len(new_series)-t+1):
				if n == 0:
					calc_ema.extend(list(np.zeros(t-1)+np.nan))
					ema_one = float(new_series.iloc[-t:-t+1,i*4:i*4+1].values)
					calc_ema.insert(0,ema_one)
				elif n !=0:
					ema_in = (calc_ema[0]*(1-k)+k*float(df.loc[counter-n-1:counter-n-1,s].values))
					calc_ema.insert(0,ema_in)
			new_series[name]= calc_ema
	return new_series

def macdext_get(df,macd,symbol, interval):#,types=False,time_period=False):
#	out_df = pd.DataFrame()
	macd_range = [[5,10,3],[10,20,7],[12,26,9],[15,35,11]]
	api = indicator_dict[macd]
	macd_ma = 1
	first=True
	for i in macd_range:
		for s in s_type:
			indicator = requests.get(api.format(symbol=symbol,interval=interval,series_type=s,fastperiod=i[0],slowperiod=i[1],signalperiod=i[2],fastmatype=ma_types[1],slowmatype=ma_types[1],signalmatype=ma_types[1]))
			time.sleep(12)
			fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
			if first:
				out_df = pd.read_csv(fixed)
				first = False
			elif first != True:
				indi_df = pd.read_csv(fixed)
				out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def stoch_get(df,stoch,symbol,interval):
	slowd = 3
	slowk = 3
	fastk = 5
	fastd = 3
	stoch_ma = 1
	#EMA
	api = indicator_dict[stoch]

	if stoch == 'stoch':
		indicator = requests.get(api.format(symbol=symbol,interval=interval,fastkperiod=fastk,slowkperiod=slowk,slowdperiod=slowd,slowkmatype=stoch_ma,slowdmatype=stoch_ma))
		time.sleep(12)
		fixed = StringIO(indicator.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		indi_df = pd.read_csv(fixed)
		return indi_df

	elif stoch == 'stochf':
		indicator = requests.get(api.format(symbol=symbol,interval=interval,fastkperiod=fastk,fastdperiod=fastd,fastdmatype=stoch_ma))
		time.sleep(12)
		fixed = StringIO(indicator.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		indi_df = pd.read_csv(fixed)
		return indi_df

def rsi_get(df,rsi,symbol,interval):
	rsi_period = [7,11,14,21]
	api = indicator_dict[rsi]
	first = True
	for t in rsi_period:
		for s in s_type:
			indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t,series_type=s))
			time.sleep(12)
			fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
			if first:
				out_df = pd.read_csv(fixed)
				first = False
			elif first != True:
				indi_df = pd.read_csv(fixed)
				out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def stochrsi_get (df,indicator,symbol,interval):
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
			time.sleep(12)
			fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
			if first:
				out_df = pd.read_csv(fixed)
				first = False
			elif first != True:
				indi_df = pd.read_csv(fixed)
				out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def adx_get(df,indicator,symbol,interval):
	api = indicator_dict[indicator]
	adx_period = [7,11,14,21]
	first = True
	for t in adx_period:
		indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t))
		time.sleep(12)
		fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		if first:
			out_df = pd.read_csv(fixed)
			first = False
		elif first != True:
			indi_df = pd.read_csv(fixed)
			out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def cci_get(df,indicator,symbol,interval):
	api = indicator_dict[indicator]
	cci_range = [5,10,15,20,35,50,65,85,100,125,200,250]
	first = True
	#annual/time period cycle high to high divided by three is the official time period
	for t in cci_range:
		indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t))
		time.sleep(12)
		fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		if first:
			out_df = pd.read_csv(fixed)
			first = False
		elif first != True:
			indi_df = pd.read_csv(fixed)
			out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def aroon_get(df,indicator,symbol,interval):
	api= indicator_dict[indicator]
	aroon_range = [5,10,15,20,35,50,65,85,100,125,200,250]
	#period since last highest high and lowest low
	first = True
	for t in aroon_range:
		indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t))
		time.sleep(12)
		fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		if first:
			out_df = pd.read_csv(fixed)
			first = False
		elif first != True:
			indi_df = pd.read_csv(fixed)
			out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def bbands_get(df,indicator,symbol,interval):
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
				time.sleep(12)
				fixed = StringIO(indicator.content.decode('utf-8'))
			#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
				if first:
					out_df = pd.read_csv(fixed)
					first = False
				elif first != True:
					indi_df = pd.read_csv(fixed)
					out_df = pd.merge(out_df,indi_df,on='time',how="inner")
	return out_df

def adosc_get(df,indicator,symbol,interval):
	api= indicator_dict[indicator]
	fastperiod = 3
	slowperiod = 10
	first = True
	#2,7?
	indicator = requests.get(api.format(symbol=symbol,interval=interval,fastperiod=fastperiod,slowperiod=slowperiod))
	time.sleep(12)
	fixed = StringIO(indicator.content.decode('utf-8'))
#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
	out_df = pd.read_csv(fixed)
	return out_df


def simple_indicator(df,indicator,symbol,interval):
	api = indicator_dict[indicator]
	indicator = requests.get(api.format(symbol=symbol,interval=interval))
	time.sleep(12)
	fixed = StringIO(indicator.content.decode('utf-8'))
#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
	out_df = pd.read_csv(fixed)
	return out_df

indicator_run = {
	'ma':moving_a,
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
		time.sleep(12)
		#response 200 = got it
		fixed = StringIO(intra_csv.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		intra_df = pd.read_csv(fixed)

		indicator_df = get_indicators(intra_df,symbol,interval)
		#don't need intra df

		out_df = pd.merge(intra_df,indicator_df,on='time',how='inner')
		return out_df
	elif interval == 'daily':
		daily_csv = requests.get(apid.format(symbol=symbol))
		time.sleep(12)
		#response 200 = got it

		#indicator_df = get_indicators(symbol,interval)
#last working position^

		fixed = StringIO(daily_csv.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
		d_df = pd.read_csv(fixed)

		indicator_df = get_indicators(d_df,symbol,interval)

#		print(d_df)
#		print(indicator_df)
		out_df = pd.merge(d_df,indicator_df,left_index=True,right_index=True,how='inner')
		return out_df

def run_live (symbol,interval):
	df = get_data(symbol,interval)
	timer = time.asctime()
	while timer[11:15] != '16:0':
		intra_json = requests.get(api.format(symbol=symbol,interval=interval))
		time.sleep(12)
		fixed = StringIO(intra_json.content.decode('utf-8'))
		intra_row_df = pd.read_csv(fixed)
		df.append(intra_row_df,ignore_index = True)
#		time.sleep(60)

def get_indicators (df,symbol,interval
#df,
):
	indicators = indicator_dict.keys()
	first = True
	for i in indicator_list:
		indi_out = indicator_run[i](df,i,symbol,interval)
		if first:
			indicator_out = indi_out
			first = False
		elif first != True:
			try:
				indicator_out = pd.merge(indicator_out,indi_out,left_index=True,right_index=True,how='inner')
			except Exception:
				print(i)
	return indicator_out

	#aroon lagging does not predict change, asesses trends strength

symbols = [#'MRVL',
#'AMD',
#'CPRX',
'601939'

]
indicator_list = [
'ma',
#'sma',
#'ema',
#'tema',
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
			name = ''.join([symbol,'_',interval_dic[interval[i]],'.tsv'])
			csv_df.to_csv(name,sep='\t')
			print('done',symbol)

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