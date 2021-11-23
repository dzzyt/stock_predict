#from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#######################GLOBAL PARAMETERS###############################################
input_tsv = ['ASML_daily.tsv',
'AMD_daily.tsv',
'MRVL_daily.tsv',
'PDS_daily.tsv',
'CPRX_daily.tsv'
#'601939_daily.tsv'
]
#input_tsv = ['AMD_daily.tsv']
targets = {'hight':[-6,-5,'high'],
'opent':[-5,-4,'open'],
'lowt':[-4,-3,'low'],
'diff_ht':[-3,-2,'diff_h'],
'diff_lt':[-2,-1,'diff_l'],
'diff_ct':[-1,None,'diff_c'],
'high':[]
}
predict_next = True
#predicts when true, tests training when false (make sure x is greater than 0 when false)
features = 510
t='hight'
target_type = [targets[t][0],targets[t][1]]
x = 0
#first x number of timestep being tested 0 means use all timesteps to train (for predicting)
batches = 1
timesteps = 501
steps = 18
################################################DATA PREPROCESSSING##################################################################
def standardized(df):
	return (df-df.mean())/df.std()
#
standard_dict = {}
df_dict ={}
run_data = {}
#input,mean,std
for f in input_tsv:
	dataf = pd.read_csv(f,sep='\t')
	dataf = dataf.iloc[:2004,2:]
	#slices start to finish index inclusive, 2 (removes duplicate index column and timestamp column) to finish columns
	for c in dataf.columns:
		if c[:4]=='time':
			#finds time columns
			dataf = dataf.drop(c,axis=1)
			#deletes a list or single column, axis means column (1) or row (0)
	diff_h= (dataf['high']-dataf['open'])/dataf['open']
	diff_l=(dataf['open']-dataf['low'])/dataf['open']
	diff_c=(dataf['open']-dataf['close'])/dataf['open']
	dataf =dataf.assign(diff_h=diff_h,diff_l=diff_l,diff_c=diff_c)
	#creates differences % columns	
	run_data[f]=[
	(dataf.iloc[:1,:]-dataf.iloc[0:timesteps,:].mean())/dataf.iloc[0:timesteps,:].std(),
	dataf.loc[0:timesteps,targets[t][2]].mean(),
	dataf.loc[0:timesteps,targets[t][2]].std()
	]
	for i in range(batches):
		name = f[:-4]+'_'+str(i)
		ndf = dataf[i*timesteps:timesteps*(i+1)]
		#STANDARDIZATION by steps 
#
	#tried raw data, seems to work
#
		#STANDARDIZATION as a whole
		#ndf = standardized(dataf)
		#currently working with one "data point"(series) with 474 511 features and ~2000 time steps
#
		standard_dict[name] = [dataf.loc[:,'high'].mean(),dataf.loc[:,'high'].std(),
		dataf.loc[:,'open'].mean(),dataf.loc[:,'open'].std(),
		dataf.loc[:,'low'].mean(),dataf.loc[:,'low'].std(),
		dataf.loc[:,'diff_h'].mean(),dataf.loc[:,'diff_l'].std(),
		dataf.loc[:,'diff_l'].mean(),dataf.loc[:,'diff_l'].std(),
		dataf.loc[:,'diff_c'].mean(),dataf.loc[:,'diff_c'].std()]
#
		ndf = standardized(ndf)
		#dictionary for de-standardizing results not complete
	#creating targets
		hight=ndf['high'].shift(1)
		opent= ndf['open'].shift(1)
		lowt = ndf['low'].shift(1)
		diff_ht= ndf['diff_h'].shift(1)
		diff_lt=ndf['diff_l'].shift(1)
		diff_ct= ndf['diff_c'].shift(1)
	#diff is open to high because low to high you can't be sure where low is 
		ndf = ndf.assign(hight=hight,opent=opent,lowt=lowt,diff_ht=diff_ht,diff_lt=diff_lt,diff_ct=diff_ct)
		ndf =ndf.drop(i*timesteps)
		df_dict[name]=ndf
#############################PREPARE INPUT DF#################################################################
columns = dataf.columns.values
run_df = pd.DataFrame(columns=columns)
for k,v in run_data.items():
	run_df = run_df.append(v[0],ignore_index=True,sort=False)
columns = ndf.columns.values
input_df = pd.DataFrame(columns=columns)
for i in range(timesteps-1):
	for k, v in df_dict.items():
		input_df = input_df.append(v[i:i+1],ignore_index=True,sort=False)
#############################################MODEL##########################################################3
class Sequence(nn.Module):
	#setting up a neural network nn.module base model for all networks as class
	def __init__(self):
		#initiates when class is called
		super(Sequence, self).__init__()
#add dropout layer?
		self.lstm1 = nn.LSTMCell(features, features)
#first lstm cell input =x, output = y (x,y) LSTM layers are fully connected... because any 0 bias should be naturally degraded
		#matrix products of all input features are inputed and bias and weights should reflect effect 
		self.lstm2 = nn.LSTMCell(features, features)
		#self.lstm2= nn.LSTMCell(timesteps,1)
#second
#		self.lstm3 = nn.LSTMCell(474, 474)
		#self.lstm3 = nn.LSTMCell(474, 1)
#third LSTM transformation of 474 to 1?
		self.linear = nn.Linear(features, 1)
		#linear cell applies linear transformation to x to return y (x,y,bias= True) why linear transformation
	def forward(self, input, future = 0):
		#forward training with data input, future for predictions for x generations
		in_size = len(df_dict)
		if len(input) == len(input_tsv):
			in_size = 1
		outputs = []
		#dynamic input size for test vs train
#initiallizing layer states all 0, why use 51 lstm layers?
		h_t = torch.zeros(in_size, features, dtype=torch.double)
		#first hidden state of first lstm in batch
		#hidden state is the output of the lstm model
		#torch.zeros creats (x,y) ndarray of zeros with shape x,y, x = batch size,y = feautures
		#input.size(0) first dimension/shape of tensor
		#dtype=double or float64 is the data type per cell
		c_t = torch.zeros(in_size, features, dtype=torch.double)
		#first cell state first lstm
		#cell state is the state retained inside the training lstm cells
		h_t2 = torch.zeros(in_size, features, dtype=torch.double)
		#first h for second lstm
		c_t2 = torch.zeros(in_size, features, dtype=torch.double)
		#first c for second lstm
#		h_t3 = torch.zeros(in_size, 474, dtype=torch.double)
#		c_t3 = torch.zeros(in_size, 474, dtype=torch.double)
		for i, input_t in enumerate(input.chunk(int(len(input)/in_size), dim=0)):
# chunking dim 0 with features in dim 1 dim 0 (in order of your sequences) features is in dim 1 (good)
#set chuncks = number of sequences or series of your input
			#given a list and optional index to start at, enumerate generate list of lists [index, item in list given]
			#given tensor, optional number of chuncks, optional dim dimension, returns chunked tensors
			#dimension 1 is horizontally through the lists
			#dimension 0 is vertically between lists
			#more dimensions to be explored!
			#.size returns (no brackets) size of tensor by dimensions, 2d = (x,y) 3d = (x,y,z)
			h_t, c_t = self.lstm1(input_t, (h_t, c_t))
			#taking input data (as list all features/values per timestap)
			#and  initialized hidden and cell state out putting hidden state and cell states from first set of lstm cells
			h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
			#taking in output of hidden states from first lstm and initialized hidden and cell states, outputting hidden and cell states from second set of lstms
			#h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
			output = self.linear(h_t2)
			#output = self.linear(h_t3)
			#output = h_t3
			#running hidden outputs from second lstm through single linear layer with output size 1 creates 1 output for that timestamp
			outputs += [output]
#        for i in range(future):# if we should predict the future
#            h_t, c_t = self.lstm1(input, (h_t, c_t))
#            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
#            output = self.linear(h_t2)
#            outputs += [output]
#            print(outputs)
		outputs = torch.stack(outputs,1).squeeze(2)
		#stack attaches the outputs into one tensor squeeze removes other dimensions need to learn more about the dimensions of how they are stacked and squeezed
		return outputs
#################################################RUN MODEL####################################################
if __name__ == '__main__':
	# set random seed to 0
	np.random.seed(0)
	torch.manual_seed(0)
	tests=x*len(df_dict)
	#makes file to train
	input = torch.tensor(input_df[tests:].values)[:,:-6]
#input data in reverse, newest to oldest
#each time step should be a tensor of n batches and m features
#sine example is 998 steps (tensors) of shape [97,1] 97 sequences and 1 feature
#	input = np.flip(input.numpy(),0).copy()
#	input = torch.from_numpy(input)
#feed in right direction oldest to newest
	target = torch.tensor(input_df[tests:].values)[:,target_type[0]:target_type[1]]
#targets in reverse
#	target = np.flip(target.numpy(),0).copy()
#	target = torch.from_numpy(target)
#feed in right direction
	target = target.chunk(int(len(input)/len(df_dict)),dim=0)
	target = torch.stack(target,1).squeeze(2)
	#target data for training where you want the next number to be
	if x >0:
		test_input = torch.tensor(input_df[:tests].values)[:,:-6]
		#testing for model analysis input
		test_target = torch.tensor(input_df[:tests].values)[:,target_type[0]:target_type[1]]
		#testing for target data for model analysis
		# build the model
	run_input = torch.tensor(run_df.values)
	seq = Sequence()
	#load model
	seq.double()
	#everything in float64
	criterion = nn.MSELoss()
	#nn.SmoothL1Loss()
	#nn.L1Loss()
	#criterion are loss functions to calculate minimizing loss
#
	#LBFGS lr default is 1
	#optim.LBFGS(seq.parameters(), lr=0.7)
	#optimizer for network helps with approximation? input the model cells and the learning rate?
	#raw data and normalized data works with lbfgs
	#normalized stepwise 25x100 does not work returns nan and clipping doesn't seem to help
	#currently the best, amazing shape and amazing loss but prediction 1 value is a little high, but still good direction might be overfitting 
	#90 steps lr 0.05 is way too low no swinging at all at 0.00026
	optimizer = optim.LBFGS(seq.parameters(), lr=0.25)
#
	#optim.RMSprop(seq.parameters(),lr=0.01,centered=True)#,momentum=0.9)
#default rmsprop 0.01
#retry with new chunking
#
	#optim.Adam(seq.parameters())
#default adam 0.001
#retry with new chunking
#
	#optim.SGD(seq.parameters(),lr=0.01,momentum=0.9,nesterov=True)
#retry with new chunking
	#begin to train
	for i in range(steps):
		#15 seems to be good number for reverse
		def closure():
			optimizer.zero_grad()
			#runs optimizer with zero gradient?
			out = seq(input)
			#runs networks with input returns out
			loss = criterion(out, target)
			#caculates loss using criterion comparing output and target
			print('loss:', loss.item())
			loss.backward()
			#what's backwards? ends training updates gradients in cells(parameters)
			#torch.nn.utils.clip_grad_value_(seq.parameters(),5)
			#clips gradients, currently not working
			return loss
		#epochs?
		print('step: ', i)
#		optimizer.zero_grad()
#		#runs optimizer with zero gradient?
#		out = seq(input)
#		#runs networks with input returns out
#		loss = criterion(out, target)
#		#caculates loss using criterion comparing output and target
#		print('loss:', loss.item())
#		loss.backward()
#		optimizer.step()
		#some optimizer don't need multiple closure per step so don't def function
		optimizer.step(closure)
#LBFGS needs multiple closure calls
############################################PREDICTION AND TESTING#####################################3
		# begin to predict, no need to track gradient here
		if predict_next == False:
			with torch.no_grad():
				#runs network without gradient updating?
				#future=1
				pred = seq(test_input)#, future=future)
	#			list_out =pred.numpy().values
	#			results_dict = {}
	#			for i in range(len(df_dict)):
	#				results_dict[i]=[]
	#			for i in range(len(list_out)):
	#				order = i%len(df_dict)
	#				new_result = results_dict[order]#.append(list_out[i])
	#				new_result.append(list_out[i])
	#				results_dict[order]= new_result
#need to export test results standardized 
		else:
			with torch.no_grad():
				pred = seq(run_input)
				print(t)
				pred = pred.numpy().tolist()[0]
				for i in range(len(input_tsv)):
					print(input_tsv[i])
					print(pred[i]*run_data[input_tsv[i]][2]+run_data[input_tsv[i]][1])

			#for raw input no de-normalization
			#print (float(pred.numpy())*std +mean)
			#denormalization for standardized input
#			print (test_target)
			#prints the actual values
#            loss = criterion(pred[:, :-future], test_target)
#            print('test loss:', loss.item())
#            y = pred.detach().numpy()
			#print (pred)
#            print(test_target)
##################################################SAVING MODELS AND RESULTS#####################################
path = ''.join([os.getcwd(),'\\funmodel.pth'])
torch.save(seq.state_dict(),path)
