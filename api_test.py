import requests
import json
import pandas as pd
from io import StringIO
import numpy as np
import time
from functools import reduce


symbol = 'ASML'
interval = 'daily'
s_type = ['close']#,'open']
api = 'https://www.alphavantage.co/query?function=EMA&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype=csv&apikey=2SXOFPK5YGV8VIVI'
apid = 'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize=full&datatype=csv&apikey=2SXOFPK5YGV8VIVI'
ma_range = [5,250]
#125
#out_df = pd.DataFrame()
def shifter(shifts,shift_df,direction='up'):
	if shifts >1:
		if direction =='up':
			output = shifter(shifts-1,shift_df,direction)
			output.append(shift_df.shift(0-shifts+1))
			return output
	elif shifts ==1:
		output = [shift_df].copy()
		return output
daily_csv = requests.get(apid.format(function = 'TIME_SERIES_DAILY',symbol=symbol))
#time.sleep(11.2)
		#response 200 = got it

		#indicator_df = get_indicators(symbol,interval)
#last working position^

fixed = StringIO(daily_csv.content.decode('utf-8'))
		#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
data = pd.read_csv(fixed)
new_series = pd.DataFrame()

################################# MA tests

#print(data.loc[1:10,'close'])

for i in range(len(ma_range)):
	for s in s_type:
		t = ma_range[i]
		name = '_'.join(['sma',str(t),s])
		new_series[name] = reduce(lambda x,y: x.add(y),shifter(t,data[s])).values/t
		name = '_'.join(['ema',str(t),s])
		counter = (len(new_series)-t+1)
		calc_ema = []
		k = 2/(t+1)
		for n in range(len(new_series)-t+1):
			if n == 0:
				calc_ema.extend(list(np.zeros(t-1)+np.nan))
				ema_one = float(new_series.iloc[-t:-t+1,i*4:i*4+1].values)
				#-t:-t+1  = -5:-4 = 5th last , 0:1 = 1
				# -250:-249 = 250th? 1:2 = 2
				calc_ema.insert(0,ema_one)
			elif n !=0:
				ema_in = (calc_ema[0]*(1-k)+k*float(data.loc[counter-n-1:counter-n-1,s].values))
				#print(data.loc[-t-n:-t-n+1,s])
				#-t -n : -t -n +1 = -6:-5 = 6th last
				calc_ema.insert(0,ema_in)
		new_series[name]= calc_ema
		indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t,series_type=s))
		fixed = StringIO(indicator.content.decode('utf-8'))
		ema_call = pd.read_csv(fixed)
		new_series = pd.merge(new_series,ema_call, left_index=True,right_index=True,how='outer')
#print (new_series.iloc[-t:-t+1,i:i+1])

################################ MA OLD
#for t in ma_range:
#	for s in s_type:
#		indicator = requests.get(api.format(symbol=symbol,interval=interval,time_period = t,series_type=s))
#		time.sleep(11.2)
#		fixed = StringIO(indicator.content.decode('utf-8'))
#	#pandas.read_csv needs a filepath for strings use StringIO from IO to convert Str to filepath
#		if first:
#			out_df = pd.read_csv(fixed)
#			first = False
#		elif first != True:			
#			indi_df = pd.read_csv(fixed)
#			out_df = pd.merge(out_df,new_series,on='time',how="inner")
#######################################


new_series.to_csv('test.tsv',sep='\t')