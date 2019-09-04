import os
import csv
import math
import h5py
import pandas as pd 
import numpy as np

from scipy.cluster.vq import kmeans,vq, whiten, kmeans2
from scipy.spatial import cKDTree, distance

from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askopenfile, askdirectory

import matplotlib
matplotlib.use("TkAgg")
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import estimate_bandwidth, MeanShift
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt

#===================================================================
#minor functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~
def h5reader3(filename1, filename2, filename3):
	"""
		-h5 file reader for storms from coastal hazards system data
		-user needs to know makeup of .h5 file (tree makeup: file1 -> file2 -> file3)
	"""
	with h5py.File(filename1, 'r') as hf:
		ret = hf.get(filename2+'/'+filename3)
		ret = np.array(ret)
	return ret

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def h5reader_prob(filename1, filename2, filename3, station):
	"""
		-h5 file reader for storms from coastal hazards system data
		-user needs to know makeup of .h5 file
	"""
	with h5py.File(filename1, 'r') as hf:
		ret = hf.get(filename2+'/'+filename3)
		ret = ret[station - 1]
	return ret

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def h5_dt_reader(file1, storm):
	with h5py.File(file1, 'r') as hf:
		hf2 = hf[storm]
		dt = hf2.attrs['Record Interval']		#reading time step
		dt = int(dt.decode("utf-8"))			#bit string to int

		dt_units = hf2.attrs['Record Interval Units']	#time step units
		dt_units = dt_units.decode("utf-8") 	#bit string to string

		if dt_units == 'sec':
			dt = dt/60							#converts seconds to minutes
	return dt

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def movingaverage(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def diff_points_elv(ts, perc_peaks):
	stormlist_sel = [i for i in ts.columns]

	peaks_x = pd.DataFrame.idxmax(ts[int(len(ts)/10) : int(len(ts)-len(ts)/10)])[0]
	avg = ts.mean(axis = 1)					#finding average curve
	mini = np.nanmin(avg)[0]
	peaks_08 = (np.nanmax(avg)[0] - mini)*perc_peaks + mini
	half = int(peaks_x)

	temp1 = ts[0:half+1]
	temp2 = ts[half:-1]
	temp2 = temp2.reset_index(drop = True)

	dbp = []
	for obj in stormlist_sel:
		temp1_r = temp1[obj][::-1]
		x1_n = []
		for i, obj2 in enumerate(temp1_r):
			count = len(temp1_r) - i
			if obj2 < peaks_08:
				if count >= len(temp1_r):
					count = count - 1
				xp = [obj2, temp1_r[count]]
				yp = [count-1, count]
				x1_n = np.interp(peaks_08, xp, yp)
				break
		if not x1_n:
			x1_n = np.nanargmin(temp1[obj])
		x2_n = []
		for i, obj2 in enumerate(temp2[obj]):
			if obj2 < peaks_08:
				if i == 0:
					i = i+1
				xp = [obj2, temp2[obj][i-1]]
				yp = [i+half, i-1+half]
				x2_n = np.interp(peaks_08, xp, yp)
				break
		if not x2_n:
			x2_n = np.nanargmin(temp2[obj]) + half

		dbp.append(x2_n - x1_n)

		x = [x1_n, x2_n]
		y = [peaks_08, peaks_08]

	return dbp

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def do_kdtree(combined_x_y_arrays,points):
	"""
	calculates the nearest value to the centroid
	"""
	mytree = cKDTree(combined_x_y_arrays)
	dist, indexes = mytree.query(points)
	return indexes

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def nan_trim(ts):
	"""
	trims the input timeseries so that there are no 'nan' values on either side of the peak
	"""
	stormlist_sel = [i for i in ts.columns]
	peaks_x = pd.DataFrame.idxmax(ts[int(len(ts)/10) : int(len(ts)-len(ts)/10)])

	nan_l = []
	nan_r = []
	for i, obj in enumerate(stormlist_sel):
		temp_l = ts[obj][0:peaks_x[obj]]
		temp_r = ts[obj][peaks_x[obj]:-1]
		nan_l_temp = temp_l.isnull().sum()
		nan_r_temp = temp_r.isnull().sum()

		nan_l.append(nan_l_temp)
		nan_r.append(nan_r_temp)

	nan_l_max = np.max(nan_l)
	nan_r_max = np.max(nan_r)+1

	ts2 = ts.ix[nan_l_max:(len(ts)-nan_r_max-1)]			#trimming matrix so there are no 'nan' values
	ts2 = ts2.reset_index(drop = True)
	
	return ts2

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def norm_storm(ts, time):
	"""
	this function normalizes a set of storms based on the average peak value
	"""
	avg = ts.mean(axis = 1)
	time = np.linspace(0, len(ts)-1, len(ts))
	time = time/max(time)									#normalizing time array

	avg_peak = np.nanmax(avg)[0]							#peak value of average
	new_storms = ts/avg_peak								#storm elevations normalized by avg. peak

	return new_storms, time

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def num_rep_storms(data):
	data_temp = whiten(data, check_finite = True)			#whitens data

	KM = [kmeans(data_temp,k) for k in range(1,4)]			#kmeans from 1 -> 3
	vari = [var for (cent,var) in KM]						#mean within-cluster sum of squares

	slope1 = vari[0] - vari[1]								#slope from frist to second point
	slope2 = vari[1] - vari[2]								#slope from second to third point

	if slope1 > 3*slope2:									#if 1st slope is greater than 3X 2nd slope, then 2 clusters are chosen
		K2_temp = 2
		centroids,_ = kmeans(data_temp,K2_temp)				#centroids from kmeans
		idx,_ = vq(data_temp,centroids)						#array of index and bin it belongs to; assign each sample to a cluster
		idx = [i for i in idx]
	else:													#otherwise 1 cluster is chosen.
		K2_temp = 1
		idx = np.zeros(len(data_temp))						#index is same for all points
		idx = [int(ii) for ii in idx]						#index to list

	return K2_temp, idx

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def num_rep_storms2(data):
	data = data[:,0].reshape(-1, 1)										#reshape for kde

	bw5 = estimate_bandwidth(data, quantile = 0.2)						#calculating bandwidth
	s5, e5, mi5, ma5 = kde_eval(data, bw5)								#performing kde

	K2_temp = 1															#initial number of sub-clusters
	idx = np.zeros(len(data))											#initial idx
	for i in ma5:
		ma5_max = max(e5[ma5])											#peak of kde
		if e5[i] != ma5_max:
			if e5[i] > 0.3*ma5_max:
				K2_temp = 2												#if peak > 0.5 max, then two sub-clusters
				idx = np.digitize(data, [0, s5[mi5[0]]])				#digitizing array between 0 and first minimum
				idx = [i[0]-1 for i in idx]
				break
			else:
				idx = np.zeros(len(data))

	return K2_temp, idx

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def kde_eval(data, bw):
	# data = data.reshape(-1,1)
	kde = KernelDensity(kernel = 'gaussian', bandwidth = bw).fit(data)
	s_temp = np.linspace(min(data), max(data), len(data))
	e_temp = np.exp(kde.score_samples(s_temp.reshape(-1,1)))
	mi2_temp, ma2_temp = argrelextrema(e_temp, np.less, order = 5)[0], argrelextrema(e_temp, np.greater, order = 5)[0]			#local min and max
	return s_temp, e_temp, mi2_temp, ma2_temp

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def split_list(n):
    """will return the list index"""
    return [(x+1) for x,y in zip(n, n[1:]) if y-x != 1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_sub_list(my_list):
    """will split the list base on the index"""
    my_index = split_list(my_list)
    output = list()
    prev = 0
    for index in my_index:
        new_list = [ x for x in my_list[prev:] if x < index]
        output.append(new_list)
        prev += len(new_list)
    output.append([ x for x in my_list[prev:]])
    return output

#~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_bins_kde(ts,bw_den):
	peaks_y = pd.DataFrame.max(ts[int(len(ts)/10) : int(len(ts)-len(ts)/10)]) 			#peak values
	peaks_y = np.array(peaks_y)															#array of peak values
	peaks_y = peaks_y.reshape(-1,1)														#reshape for kde

	bw = estimate_bandwidth(peaks_y, quantile = 0.1)/bw_den								#initial bandwidth
	kde = KernelDensity(kernel = 'gaussian', bandwidth = bw).fit(peaks_y)				#performing kde
	s = np.linspace(min(peaks_y), max(peaks_y), len(peaks_y))							#'x'-values of kde
	e = np.exp(kde.score_samples(s.reshape(-1,1)))										#'y'-values of kde
	mi, ma = argrelextrema(e, np.less, order = 5)[0], argrelextrema(e, np.greater, order = 5)[0]	#local and max and min within 5 pts. 


	#~~~~~~~~~~~
	#placing peaks into clusters
	mi_old = 0																			#initial lower value of cluster limit
	temp_data_stg = []																	#space for peak lists 
	for i, obj in enumerate(mi):
		temp = np.where((peaks_y>mi_old) & (peaks_y<s[obj]))[0]							#index of peaks in clusters
		temp_data_stg.append(peaks_y[temp])												#appending data
		if obj == mi[-1]:																#if last value in list
			temp2 = np.where(peaks_y>s[obj])[0]
			temp_data_stg.append(peaks_y[temp2])

		mi_old = s[obj]																	#updating lower value

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	""" performing kde again with lower bandwidth for 
		high density areas. gets tough to explain. """

	lrg_ma = np.where(e[ma] > max(e)/2)[0]												#where initial kde local maxima are greater than half of total kde max
	lrg_ma2 = get_sub_list(lrg_ma)														#creates sub lists of consectutive numbers (e.g. [[1,2,3], [5,6]] )
	mi2, ma2, s2 = [], [], []															#setting aside space
	for ii in lrg_ma2:																	#loop through outer list in lrg_ma2
		new_data = []
		for i in ii:																	#creating list of peaks within inner list of lrg_ma2
			new_data.append(temp_data_stg[i].tolist())

		new_data = [item for sublist in new_data for item in sublist]					#flatten list
		new_data = np.array(new_data)													#reshaping for kde
		new_data = new_data.reshape(-1,1)

		kde2 = KernelDensity(kernel = 'gaussian', bandwidth = bw/2).fit(new_data)		#performing kde
		s_temp = np.linspace(min(new_data), max(new_data), len(new_data))				#'x' values
		e_temp = np.exp(kde2.score_samples(s_temp.reshape(-1, 1)))						#'y' values
		mi2_temp, ma2_temp = argrelextrema(e_temp, np.less, order = 5)[0], argrelextrema(e_temp, np.greater, order = 5)[0]			#local min and max
		mi2.append(mi2_temp)															#appends to list
		ma2.append(ma2_temp)
		s2.append(s_temp[mi2_temp])														#values associated with minima. (e.g. 1.5m)

	mi = np.delete(mi, lrg_ma[:-1])														#removes cluster mins from original min array
	s_mi_new = np.sort(np.append(s[mi], s2))											#creates new array with all minimumums

	#~~~~~~~~~~~~
	""" check for clusters with too many or not enough storms. """
	idx = np.digitize(peaks_y, s_mi_new)												#digitizing array
	idx = [i[0] for i in idx]															#to list
	idx_count = [idx.count(i) for i in range(0, max(idx)+1)]							#counting storms in each bin

	new_list = s_mi_new[:]

	count = 0																			#setting up counter
	while (max(idx_count) > 50) or (min(idx_count) < 5):								#while loop for if more than 50 storms, or less than 5 in cluster
		
		if max(idx_count) > 50:
			temp_idx = np.where( np.array(idx_count) > 50)[0]							#index of where there are greater than 50 storms
			new_values = []																#setting aside space for new bin limits
			for i in temp_idx:
				bin_avg = (new_list[i] + new_list[i-1])/2								#getting average of bin values to add an additional bin to list
				if i == 0:
					bin_avg = (new_list[i] + 0)/2
				new_values.append(bin_avg)												#appending new bin values to new_values

			new_list = np.sort(np.append(new_list, new_values))							#adding and appending new values
			idx = np.digitize(peaks_y, new_list)										#placing peaks in bins
			idx = [i[0] for i in idx]													#to lsit
			idx_count = [idx.count(i) for i in range(0, max(idx)+1)]					#recounting peaks in each cluster
		
		if min(idx_count) < 5:
			temp_idx = np.where( np.array(idx_count) < 5)[0]							#index of where there are less than 5 storms in a cluster
			new_values = new_list[:]													#list of cluster limits
			for i in temp_idx:
				if i == len(new_list):													#if there are <5 storms in last cluster
					new_values = np.delete(new_values, np.where(new_values == new_list[-1])[0])		#combines last two clusters
				else:
					new_values = np.delete(new_values, np.where(new_values == new_list[i])[0])		#otherwise, combines with the next cluster. 
			
			new_list = new_values[:]
			idx = np.digitize(peaks_y, new_list)										#placing peaks in bins
			idx = [i[0] for i in idx]													#to list
			# idx_count = [idx.count(i) for i in range(0, max(idx)+1)]

		idx_count = [idx.count(i) for i in range(0, max(idx)+1)]						#recounting peaks in each cluster
		count = count + 1																#increasing the counter
		if count > 5:																	#exiting while loop if counter exceeds 5
			break

	#~~~~~~~~~~~~
	""" prepping for output variables """
	K = len(new_list) + 1																#number of clusters
	bin_sorted = [i for i in range(0, K)]												#the clusters, sorted (e.g. [0,1,2,..] )
	idx = np.digitize(peaks_y, new_list)												#index of peaks in each cluster

	peak_val_max = [i for i in new_list[1:]]											#upper cluster limits
	peak_val_min = [i for i in new_list[:-1]]											#lower cluster limits

	bin_range = np.column_stack((np.array(peak_val_min), np.array(peak_val_max)))		#stacking lower and upper limits

	return bin_sorted, idx, K, bin_range

#===================================================================


#===================================================================
#major functions
def loading_data(input_path_full, input_csv, ft_m):
	temp = os.path.split(input_path_full)									#splitting path and .h5
	input_h5 = temp[-1]														#input .h5
	input_path_h5 = temp[0]													#input path to .h5

	os.chdir(input_path_h5)

	with h5py.File(input_h5, 'r') as hf:
		all_storms = list(hf.keys())										#reading all storm names
		station_id = str(hf.attrs['Save Point ID']).split('.')[0]			#reading save point id
		project_id = str(hf.attrs['Project'].decode('utf-8'))				#reading proejct (NACCS vs. S2G)
		et_t = str(hf[all_storms[0]].attrs['Storm Type'].decode('utf-8'))	#reading etrop or trop

	if input_csv is None:													#check for csv containing select storms (200km)
		stormlist_sel = all_storms[:]
	else:
		stormlist_sel = input_csv[:]

	#~~~~~~~~~~~~~~~~~~~~~~~~~~

	ts = pd.DataFrame()														#creating space for data (dataframes and list)
	dt = pd.DataFrame()
	column_name = []
	for i, obj in enumerate(stormlist_sel):									#reading storms from h5 file in 'stormlist_sel'
		file1 = input_h5 													#'tree' values in h5
		file2 = obj
		file3 = 'Water Elevation'
		file_t = 'yyyymmddHHMM'
		print(file2)
		ts1 = h5reader3(file1, file2, file3)								#reading elevation data
		ts1 = pd.DataFrame(ts1)												#create dataframe
		ts = pd.concat([ts, ts1], ignore_index = True, axis = 1)			#appending to ts

		dt_temp = h5_dt_reader(input_h5, obj)								#reading time step for storm			
		dt_temp = pd.Series(dt_temp)										#create dataframe
		dt = pd.concat([dt, dt_temp], ignore_index = True, axis = 1)		#append to dt

		temp = obj.split()													#creating column names
		column_name.append(temp[0])											#append to column_name
	
	ts.columns = column_name												#adding column names to dataframe
	dt.columns = column_name

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	#making sure timeseries aren't nan
	peaks_y = pd.DataFrame.max(ts[int(len(ts)/10) : int(len(ts)-len(ts)/10)])	#getting peak data
	storm_new = np.argwhere(np.isfinite(peaks_y))								#where peak data is finite
	storm_new = [i for sublist in storm_new for i in sublist]					#new storm list, sublists to one list
	stormlist_sel = [column_name[i] for i in storm_new]							#new storm list

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	#getting rid of data with "-99999999"
	stormlist_sel_c = stormlist_sel[:]			#a copy of stormlist_sel; stormlist_sel changed in loop; don't want to mess up loop
	for obj in stormlist_sel_c:
		temp = ts[obj]
		temp_min = temp.min()
		if temp.min() < -100:
			stormlist_sel.remove(obj)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	ts = ts[stormlist_sel]						#final timeseries data
	dt = dt[stormlist_sel]						#final dt data

	if ft_m == True:							#conversion if necessary
		ts = ts/.3048

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	return ts, dt, station_id, project_id, et_t



#===================================================================



def organizing_data(ts, dt, smooth):
	stormlist_sel = [i for i in ts.columns]			#list of stormnames from column headers

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	#smoothing; if applicable
	if smooth == True:
		for i in stormlist_sel:
			ts_start = ts[i].iloc[0]				#start and end points; to retain length of array
			ts_last = ts[i].iloc[-1]
			ts[i] = movingaverage(ts[i], 3)
			ts[i].iloc[0] = ts_start
			ts[i].iloc[-1] = ts_last

	peaks_x = pd.DataFrame.idxmax(ts[int(len(ts)/10) : int(len(ts)-len(ts)/10)])		#index of peaks. looks at inner 80%
	peaks_x_max = pd.DataFrame.max(peaks_x)												#maximum index of peaks; for shift

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	#===============================================================
	#maybe work on this some more..... filtering noisy data.
		# stormlist_sel_c = stormlist_sel[:]
		# del_storm2 = []
		# peaks_y = pd.DataFrame.max(ts[int(len(ts)/10) : int(len(ts)-len(ts)/10)])			#value of peaks; inner 80%

		# for i, obj in enumerate(stormlist_sel_c):
		#     data = ts[obj]
		#     data = data[~np.isnan(data)]
		#     diff = np.diff(data)
		#     diff2 = np.diff(diff)
		#     diff_sum = np.cumsum(abs(diff2))

		#     if diff_sum[-1] > 3*peaks_y[obj]:
		#         del_storm2.append(obj)
		#         ts = ts.drop(obj, 1)
		#         stormlist_sel.remove(obj)
	#===============================================================
	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	#shifts time series so peaks align by inserting 'NaN' values
	temp_df = pd.DataFrame()
	for i, obj in enumerate(stormlist_sel):
		shift_idx = int(peaks_x_max -peaks_x[obj])								 #shift index

		temp = ts[obj]															#temporary time series
		nan_a = np.zeros(shift_idx)+np.nan 										#creating array of 'NaN' values
		temp2 = pd.DataFrame(np.append(nan_a, temp))							#dataframe of nan and temp. time series
		temp_df = pd.concat([temp_df, temp2], ignore_index = True, axis =1) 	#appending to temporary dataframe

	ts = temp_df[:]
	ts.columns = stormlist_sel

	time = pd.DataFrame()														#creating dataframe of time for shifted plots
	for obj in stormlist_sel:
		time_temp = np.linspace(0, int(len(ts[obj])*dt[obj]-dt[obj]), len(ts[obj]))
		time_temp = pd.DataFrame(time_temp)
		time = pd.concat([time, time_temp],  ignore_index = True, axis = 1)

	time.columns = stormlist_sel
	# plt.figure()
	# plt.plot(time/1440, ts, lw = 0.75)
	# plt.grid()
	# plt.xlabel('Time (days)')
	# plt.ylabel('Surge (m. MSL)')
	# plt.show()
	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	return ts, time



#===================================================================



def create_bins(ts, user_bin, kde_TF, bw_est):
	stormlist_sel = [i for i in ts.columns]
	peaks_y = pd.DataFrame.max(ts[int(len(ts)/10) : int(len(ts)-len(ts)/10)])

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	#elbow method, if grouping by kmeans
	if kde_TF == True:
		bin_sorted, idx, K, bin_range = create_bins_kde(ts, bw_est)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	#if cluster limits are defined already
	else:
		K = len(user_bin)
		bin_sorted = [i for i in range(0, K)]

		user_bin_temp = [i[0] for i in user_bin]
		user_bin_temp.append(user_bin[-1][1] + 0.0000001)
		user_bin_temp = np.array(user_bin_temp)
		idx = np.digitize(peaks_y, user_bin_temp) - 1 	#Returns which bin each storm belongs to

		peak_val_max = [i for i in user_bin[1:]]
		peak_val_min = [i for i in user_bin[:-1]]

		bin_range = user_bin[:]
	#~~~~~~~~~~~~~~~~~~~~~~~~~~


	return bin_sorted, idx, K, bin_range



#===================================================================



def k_means(ts, time, bin_sorted, idx, ft_m_lbl):
	#filters selected storms, and selects representative storm
	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	#setting up variables and setting aside space
	stormlist_sel = [i for i in ts.columns]
	peaks_y = pd.DataFrame.max(ts[int(len(ts)/10) : int(len(ts)-len(ts)/10)])

	K2 = []
	rep_storms = []
	idx2_stg = []
	s_n_r_stg = []

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	for i in bin_sorted:											#loop through bins in order
		if i in idx:												#check for storms in bin
			a = np.argwhere(idx==i)
			temp_storms = [stormlist_sel[ii[0]] for ii in a]		#list of storms in bin
			ts_temp = ts[temp_storms]								#time series of storms in bin
			time_temp = time[temp_storms]
			peaks_select = peaks_y[temp_storms]						#peaks of storms in bin
			
			ts_temp_norm, _ = norm_storm(ts_temp, time_temp)	#normalizing storms
			ts_temp_norm = nan_trim(ts_temp_norm)				#removing nan values

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	#if > 10 storms in bin, only looks at middle 60%
			if len(temp_storms) >= 10:
				peaks_sel_rng = pd.Series.max(peaks_select) - pd.Series.min(peaks_select)			#peak range (highest - lowest)
				peaks_u20 = .8*peaks_sel_rng + pd.Series.min(peaks_select)							#upper 80% value
				peaks_l20 = .2*peaks_sel_rng + pd.Series.min(peaks_select)							#lower 20% value

				s_n_r_idx = np.argwhere((peaks_select > peaks_l20) & (peaks_select < peaks_u20))	#storms in range index (middle 60%)
				s_n_r_idx = [ii[0] for ii in s_n_r_idx]												#s_n_r_idx to list

				if len(s_n_r_idx) == 1:										#if there is only one storm in middle 60%, selects storm as
					stormlist_rep = temp_storms[s_n_r_idx[0]]				#	representative storm
					K2_temp = 1												#number of clusters
					idx2 = np.zeros(len(temp_storms))						#zeros, for idx2
					idx2 = [int(ii) for ii in idx2]							#list form
				
			else:
				s_n_r_idx = [ii for ii in range(len(temp_storms))]			#if < 10 storms, looks at all storms

	#~~~~~~~~~~~~~~~~~~~~~~
	#filter incomplete storms, suggest 1/2 subclusters
		#~~~~~~~~~~~~~~~~~~~~~~~~~
			#check for incomplete hydrographs
			if len(s_n_r_idx) >= 1:
				temp_storms_2 = [temp_storms[ii] for ii in s_n_r_idx]	#storms that made it through initial filter
				min_y = []												#setting aside space for lowest value of each time series

				for ii, obj in enumerate(temp_storms_2): 				#loop through each storm in bin
					temp = pd.DataFrame.min(ts_temp[obj])				#selects lowest value of time series
					min_y.append(temp)

				peaks_y2 = [peaks_select[ii] for ii in temp_storms_2]	#peak values of storms through first filter
				diff_y = np.array(peaks_y2)-np.array(min_y)				#difference b/w min and max
				dy50 = min_y + (diff_y*0.7) 							#min_y + 70% of difference b/w min and max; time series must pass through here on both sides 
				
				stormlist_sel_c = temp_storms_2[:]						#copy of temp_storms_2
				s_n_r_idx_c = s_n_r_idx[:]							 	#copy of s_n_r_idx
				for ii, obj in enumerate(stormlist_sel_c):
					half = pd.Series.idxmax(ts_temp[obj][int(len(ts_temp[obj])/10) : int(len(ts_temp[obj])-len(ts_temp[obj])/10)])		#midpoint of timeseries, index of peak value
					temp1 = ts_temp[obj][0:half]						#first half (before peak)
					temp2 = ts_temp[obj][half:-1]						#second half (after peak)
					a1 = (temp1 - dy50[ii])								#first half - delimiting elevation
					a2 = (temp2 - dy50[ii])								#second half - delimiting elevation
					a1 = pd.DataFrame.min(a1)							#minimum value of a1
					a2 = pd.DataFrame.min(a2)							#minimum value of a2

					if (a1 > 0) or (a2 > 0):							#check for postive. if positive, means that time series half didn't pass through delimiting value
						temp_storms_2.remove(obj)						#removes from temp_storms_2
						s_n_r_idx_c.remove(s_n_r_idx[ii])				#removes from s_n_r_idx

				if len(s_n_r_idx_c) < 1:								#if all storms are removed, then it restores s_n_r_idx to before this check
					s_n_r_idx_c = s_n_r_idx[:]

				s_n_r_idx = s_n_r_idx_c[:]								#after check for incomplete copies s_n_r_idx back to original name
				
				#~~~~~~~~~~~~~~~~~~~~~~~~~
				#if > 10 storms in bin, performs check for one or two representative storms
				if len(temp_storms) >= 10:
					ts_temp_norm, _ = norm_storm(ts_temp, time_temp)	#normalizing storms
					ts_temp_norm = nan_trim(ts_temp_norm)				#removing nan values

					#horizontal difference between points at percentage of avg. elevation time series
					dbp5f = diff_points_elv(ts_temp_norm, .5)
					dbp7f = diff_points_elv(ts_temp_norm, .7)
					data_temp2_full = np.column_stack((dbp5f, dbp7f))	#used to decide 1 or two storms; stacking difference between points arrays
					K2_temp, idx2 = num_rep_storms2(data_temp2_full)		#returns 1 or 2 storms, and index of which sub-cluster each storm belongs to 
					
				else:													#if < 10 storms, only 1 storm in subcluster
					K2_temp = 1
					idx2 = np.zeros(len(temp_storms))
					idx2 = [int(ii) for ii in idx2]

				#~~~~~~~~~~~~~~~~~~~~~~~~~
				#check for less than 5 storms in sub-clusters, if applicable (2 sub-clusters)
				if K2_temp == 2:
					l = list(idx2)
					count_1 = l.count(0)								#counts number of 0's in idx
					count_2 = l.count(1)								#counts number of 1's in idx

					if count_1 < 5:										#if < 5 storms in sub-cluster
						K2_temp = 1										#	one sub-cluster
						idx2 = np.zeros(len(temp_storms))
						idx2 = [int(ii) for ii in idx2]
					elif count_2 < 5:
						K2_temp = 1
						idx2 = np.zeros(len(temp_storms))
						idx2 = [int(ii) for ii in idx2]

			K2.append(K2_temp)
			s_n_r_stg.append(s_n_r_idx)
			idx2_stg.append(idx2)
	#~~~~~~~~~~~~~~~~~~~~~~~~~
	#allows user to view and over-ride number of subclusters in a cluster (i.e 2 subclusters in cluster 3; long v. short)
	K2_new = select_num_storms(ts, time, K2, idx, idx2_stg, bin_sorted, ft_m_lbl).get_list()
	K2_new = [int(i) for i in K2_new]									#to list
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~
	#selects representative storms
	for i, obj in enumerate(bin_sorted):							#loop through bins
		if obj in idx:												#check if storms are in bin
		#~~~~~~~~~~~~~~~~~~~~~~~~~
			#setting up variables
			a = np.argwhere(idx==obj)								#index of storms in bin
			temp_storms = [stormlist_sel[ii[0]] for ii in a]		#storm names in bin
			ts_temp = ts[temp_storms]								#storms in bin
			time_temp = time[temp_storms]
			peaks_select = peaks_y[temp_storms]						#peaks in bin
			s_n_r_idx = s_n_r_stg[i]								#storms that made it through filters above
			K2_new_temp = K2_new[i]									#new list of subclusters

			temp_storms_snr = [temp_storms[ii] for ii in s_n_r_idx]	#storm names that got through filter above
			ts_temp_snr = ts_temp[temp_storms_snr]					#storm time series that got through filter
			time_temp_snr = time_temp[temp_storms_snr]
			peaks_select_snr = peaks_select[temp_storms_snr]		#peaks that got through filter above

			ts_temp, time_temp = norm_storm(ts_temp_snr, time_temp_snr)		#normalizing storms
			ts_temp = nan_trim(ts_temp)								#trimming storms to get rid of nan values

			#horizontal ...
			dbp5f = diff_points_elv(ts_temp, .5)
			dbp6f = diff_points_elv(ts_temp, .6)
			dbp7f = diff_points_elv(ts_temp, .7)

			#and vertical.
			peaks_x2 = pd.DataFrame.idxmax(ts_temp)
			pts_l = peaks_x2[0]
			pts_r = len(ts_temp) - peaks_x2[0]
			pts_min = min((pts_l, pts_r))

			d_00l = ts_temp.loc[[0]].T
			d_25l = ts_temp.loc[[int(pts_min/4)]].T
			d_00r = ts_temp.loc[[peaks_x2[0] + pts_min-1]].T
			d_25r = ts_temp.loc[[peaks_x2[0] + int(3*pts_min/4)]].T

			#~~~~~~~~~~~~~~~~~~~~~~~~~~
			#stacking data
			data_full = np.column_stack((peaks_select_snr, peaks_select_snr, dbp5f, dbp6f, dbp7f, d_00l, d_25l, d_00r, d_25r))
			data_temp2_full = np.column_stack((dbp5f, dbp7f))
			
		#~~~~~~~~~~~~~~~~~~~~~~~~~~
			#determines which subcluster storms fall into. 
			#creates 2 different idx2 lists; one with filtered storms (idx2), one with all storms (idx2_new)
			if K2_new[i] == 2:
				centroids,_ = kmeans(data_temp2_full, K2_new_temp)		#centroids from kmeans
				idx2,_ = vq(data_temp2_full, centroids)					#array of index and bin it belongs to; assign each sample to a cluster
				idx2 = [ii for ii in idx2]								#index of short and long storms (0's and 1's)
				a0 = np.argwhere(np.array(idx2) == 0)					#finding storms in each subcluster
				a1 = np.argwhere(np.array(idx2) == 1)					#	(which storms are long, which are short)
				temp_storms_0 = [temp_storms_snr[ii[0]] for ii in a0]	#long vs. short storm names
				temp_storms_1 = [temp_storms_snr[ii[0]] for ii in a1]
				nis = [obj for obj in temp_storms if obj not in temp_storms_snr]	#storms that got filtered out
				ts_temp_avg0 = ts[temp_storms_0].mean(axis = 1)			#average of long and short storms
				ts_temp_avg1 = ts[temp_storms_1].mean(axis = 1)

				idx2_new = pd.DataFrame(idx2).T 						#idx2 to dataframe; to keep data organized better by
				idx2_new.columns = temp_storms_snr						#	assigning column names to each value in idx2; will contain all storms

				for ii in nis:
					ts_temp_nis = ts[ii]								#time series that got filtered out
					ts_diff0 = pd.DataFrame(ts_temp_avg0.values - ts_temp_nis.values)	#difference between above and average of both long and short
					ts_diff1 = pd.DataFrame(ts_temp_avg1.values - ts_temp_nis.values)
					
					ts_diff0avg = abs(ts_diff0.mean())					#absolute average value of new timeseries; one value
					ts_diff1avg = abs(ts_diff1.mean())
					
					if ts_diff0avg[0] < ts_diff1avg[0]:					#finds which absolute average is smaller
						idx2_new[ii] = pd.Series(0, index = idx2_new.index)		#creates new column in idx2 with elinated storms assigned to 0 or 1. 
					elif ts_diff1avg[0] < ts_diff0avg[0]:
						idx2_new[ii] = pd.Series(1, index = idx2_new.index)

				idx2_new = idx2_new.sort_index(axis = 1)				#sorts columns based on column name
				idx2_new = idx2_new.values.tolist()[0]					#converts dataframe back to list

			else:
				idx2 = np.zeros(len(temp_storms_snr))					#filtered storms index; if one subcluster, only use 0's in idx2
				idx2 = [int(ii) for ii in idx2]

				idx2_new = np.zeros(len(temp_storms))					#all storms index
				idx2_new = [int(ii) for ii in idx2_new]
			
			idx2_stg[i] = idx2_new 										#idx2 storage; all storms

		#~~~~~~~~~~~~~~~~~~~~~~~~~~	
			#loops through each subcluster. selects representative storm from centroid of data
			#	considers only storms that made it through filter above
			data_full = whiten(data_full, check_finite = True)		#whitens data for kmeans
			stormlist_rep = []
			for j in range(0,K2_new_temp):							#loop through subclusters
				j = int(j)											#subcluster number (0 or 1)
				idx2_temp = np.argwhere(np.array(idx2) == j)		#finds index of storms in subcluster
				idx2_temp = [ii[0] for ii in idx2_temp]
				temp_storms2 = [temp_storms_snr[ii] for ii in idx2_temp]	#filtered storm names

				data_full_temp = [data_full[ii, :] for ii in idx2_temp]	#data in subclusters
				centroids, _ = kmeans(data_full_temp, 1)				#kmeans, returns centroid
				nearest_idx = do_kdtree(data_full_temp, centroids)		#finds nearest value to centroid; index
				storm_temp = [temp_storms2[ii] for ii in nearest_idx]	#finds storm name associated with nearest value above
				stormlist_rep.append(storm_temp[0])						#appends to inner list of storms

			rep_storms.append(stormlist_rep)						#appends to outer list of representative storms

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	return rep_storms, K2_new, idx2_stg, bin_sorted





#=============================================
#interactive windows. 

#~~~~~~~~~~~~~~~~~~~~~~~~~~
class Select_clusters:
	def __init__(self, K_plt, vari):
		self.new = Toplevel()												#new window
		self.frame = Frame(self.new, bd = 1, relief = RAISED)				#creating frames in new window
		self.frame2 = Frame(self.new, bd = 1, relief = RAISED)

		self.new.grid_columnconfigure(1, weight = 1)						#stretches frames to fill windows
		self.new.grid_columnconfigure(2, weight = 1)
		self.new.grid_columnconfigure(3, weight = 1)
		self.new.grid_columnconfigure(4, weight = 1)
		self.new.grid_columnconfigure(5, weight = 1)

		self.new.grid_rowconfigure(1, weight = 1)
		self.new.grid_rowconfigure(2, weight = 1)
		self.new.grid_rowconfigure(3, weight = 1)
		self.new.grid_rowconfigure(4, weight = 1)
		self.new.grid_rowconfigure(5, weight = 1)

		self.frame.grid(row = 5, column = 1,  columnspan = 4, sticky=E+W)	#location of frames
		self.frame2.grid(row = 1, column = 1, rowspan = 4, columnspan = 4, sticky=N+S+E+W)

		self.fig = Figure()													#creating figure
		self.ax = self.fig.add_subplot(111)
		self.canvas = FigureCanvasTkAgg(self.fig, master = self.frame2)
		self.canvas.show()
		self.canvas.get_tk_widget().pack(side = 'bottom', fill = 'both', expand = 1)	#location of figure

		self.ax.plot(K_plt, vari, 'k.')										#plotting elbow method
		self.ax.grid(True)
		self.ax.set_title('Distortion vs. Number of Clusters')
		self.ax.set_xlabel('Number of Clusters')
		self.ax.set_ylabel('Distortion')

		self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.frame2)	#zoom, pan, etc. toolbar
		self.toolbar.update()
		self.toolbar.pack(side = TOP)

		self.entr = ttk.Entry(self.frame)									#place to input number of clusters
		self.entr.grid(column = 2, row = 2)
		self.b2 = Button(self.frame, text = 'OK', command = self.op2)
		self.b2.grid(column = 3, row = 2)
		self.l2 = Label(self.frame, text = 'Select Number of Clusters')
		self.l2.grid(column = 1, row = 1)
		self.new.wait_window(self.frame)									#wait to continue execution until window is closed

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def op2(self):
		self.groups = self.entr.get()										#getting number of clusters from input above
		self.frame.destroy()												#destroying frame and windows
		self.frame2.destroy()

		self.new.destroy()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def __int__(self):
		self.groups = int(self.groups)
		return self.groups										#returning number of clusters


#~~~~~~~~~~~~~~~~~~~~~~~~~~
class select_num_storms:
	#~~~~~~~~~~~~~~~~~~~~~~~~~~
		#setting up window and variables
	def __init__(self, ts, time, K2, idx, idx2, bin_sorted, ft_m_lbl):
		self.ts = ts
		self.time = time
		self.K2 = K2
		self.K2_o = K2[:]
		self.idx = idx
		self.idx2 = idx2
		self.bin_sorted = bin_sorted
		self.ft_m_lbl = ft_m_lbl
		self.stormlist_sel = [i for i in self.ts.columns]

		self.new = Toplevel()												#new window
		self.frame = Frame(self.new, bd = 1, relief = RAISED)				#creating frames in new window
		self.frame2 = Frame(self.new, bd = 1, relief = RAISED)
		self.frame3 = Frame(self.new, bd = 1, relief = RAISED)

		self.opt = range(1, len(K2)+1)										#dropdown menu options
		self.opt = [str(i) for i in self.opt]								#to list

		self.vari = StringVar(self.frame)
		self.drop_dwn1 = OptionMenu(self.frame, self.vari, *self.opt, command = self.bin_num)		#dropdown menu
		self.drop_dwn1.grid(row = 1, column = 1)							#location of dropdown menu

		self.entr1 = ttk.Entry(self.frame)									#entry box
		self.entr1.grid(row = 1, column = 2)								#entry box location

		self.btn1 = ttk.Button(self.frame, text = 'Save Selected Number of Sub-Clusters', command = self.save_storm)		#button to save number of sub-clusters
		self.btn1.grid(row = 1, column = 6)									#button location

		self.lbl = ttk.Label(self.frame, text = '')							#empty label for if value other than one or two is selected
		self.lbl.grid(row = 1, column = 7)									#location of label

		self.btn2 = ttk.Button(self.frame2, text = 'Continue', command = self.cont)		#button to continue to next screen (reviewing selected storms)
		self.btn2.grid(row = 4, column = 1)												#location of button

		#~~~~~~~~~~~~~~~~~~~~~~~~~~
			#creating table of subcluster summary
		self.tbl1 = ttk.Treeview(self.frame2, columns = ('Cluster', 'Number of Sub-Clusters', 'Number of Storms in Cluster'), height = len(self.K2))		#three columns
		self.tbl1.heading('#1', text = 'Cluster')
		self.tbl1.heading('Number of Sub-Clusters', text = 'Number of Sub-Clusters')
		self.tbl1.heading('Number of Storms in Cluster', text = 'Number of Storms in Cluster')
		self.tbl1.column('#1', stretch= YES)
		self.tbl1.column('Number of Sub-Clusters', stretch = YES)
		self.tbl1.column('Number of Storms in Cluster', stretch = YES)
		self.tbl1['show'] = 'headings'
		self.tbl1.grid(row = 1, column = 1, rowspan = 3)

		for i, obj in enumerate(self.K2):		#writing to table
			self.tbl1.insert('', 'end', text = ' ', values = (self.opt[i], str(obj), str(len(self.idx2[i])) ))
		#~~~~~~~~~~~~~~~~~~~~~~~~~~
			#creating space for figure
		self.fig = Figure()
		self.ax = self.fig.add_subplot(111)
		self.canvas = FigureCanvasTkAgg(self.fig, master = self.frame3)
		self.canvas.show()
		self.canvas.get_tk_widget().pack(side = 'bottom', fill = 'both', expand = 1)		#stretch to fill window
		self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.frame3)	#zoom, pan, etc. options
		self.toolbar.update()
		self.toolbar.pack(side = TOP)

		self.new.grid_columnconfigure(1, weight = 1)						#stretch to fill window
		self.new.grid_columnconfigure(2, weight = 1)
		self.new.grid_columnconfigure(3, weight = 1)
		self.new.grid_columnconfigure(4, weight = 1)
		self.new.grid_columnconfigure(5, weight = 1)

		self.new.grid_rowconfigure(1, weight = 1)
		self.new.grid_rowconfigure(2, weight = 1)
		self.new.grid_rowconfigure(3, weight = 1)
		self.new.grid_rowconfigure(4, weight = 1)
		self.new.grid_rowconfigure(5, weight = 1)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~
			#frame locations in window
		self.frame.grid(row = 5, column = 1,  columnspan = 4, sticky=E+W)
		self.frame2.grid(row = 1, column = 5, rowspan = 5, sticky=N+S)
		self.frame3.grid(row = 1, column = 1, rowspan = 4, columnspan = 4, sticky=N+S+E+W)
		self.new.wait_window(self.frame)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
		#viewing each cluster
	def bin_num(self, value):
		if self.lbl:														#gets rid of label if present
			self.lbl.destroy()
			
		#displays the storms in a given bin, and allows user to choose new number of sub-clusters
		self.ax.cla()														#clearing plot
		self.value_1 = int(value) - 1										#cluster number from dropdown menu
		a = np.argwhere(self.idx==self.bin_sorted[self.value_1])			#index of storms in bin		
		self.temp_storms = [self.stormlist_sel[ii[0]] for ii in a]			#storm names of storms in bin
		self.ts_temp = self.ts[self.temp_storms]							#elevation time series of storms in sub-bin
		self.time_temp = self.time[self.temp_storms]						#time series of storms in sub-bin
		
		#clears 'Save storms' button, and sets up new button
		if self.btn1:					
			self.btn1.destroy()
		self.btn1 = ttk.Button(self.frame, text = 'Save Selected Number of Sub-Clusters', command = self.save_storm)
		self.btn1.grid(row = 1, column = 6)

		#plots storms in sub-bin, k-means selected storm, and user selected storms
		if max(self.idx2[self.value_1]) == 0:								#plots same color if one subcluster recommended
			self.ax.plot(self.time_temp, self.ts_temp,'k', linewidth = 1)

		elif max(self.idx2[self.value_1]) > 0:								#plots different colors if two subclusters recommended
			b0 = np.argwhere(np.array(self.idx2[self.value_1]) == 0)		#index of one set of storms
			b1 = np.argwhere(np.array(self.idx2[self.value_1]) == 1)		#index of other set of storms
			
			temp_storms0 = [self.temp_storms[ii[0]] for ii in b0]			#storm names of storms in sub-bin
			temp_storms1 = [self.temp_storms[ii[0]] for ii in b1]			#storm names of storms in sub-bin

			self.ax.plot(self.time_temp[temp_storms0], self.ts[temp_storms0], 'k', linewidth = 1)
			self.ax.plot(self.time_temp[temp_storms1], self.ts[temp_storms1], 'r', linewidth = 1)

		self.ax.grid(True)			#plotting options (grid, title, axis labels)		
		self.ax.set_title('Cluster Number: %s\nNumber of storms in Cluster: %s\nAlgorithm Recommended Number of Sub-Clusters: %s' %(value, len(a), self.K2_o[self.value_1]))
		self.ax.set_xlabel('Time (days)')
		self.ax.set_ylabel('Elevation (%s)' %self.ft_m_lbl)
		self.canvas.draw()

		self.toolbar.update()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
		#saving number of subclusters if changed
	def save_storm(self):
		self.K2[self.value_1] = self.entr1.get()							#getting value from entry
		if self.lbl:														#destroy label if present
			self.lbl.destroy()

		if (int(self.K2[self.value_1]) < 1) or (int(self.K2[self.value_1]) > 2):		#check if value is 1 or 2
			self.lbl = ttk.Label(self.frame, text = 'Must Select 1 or 2 Sub-Clusters')	#prints error next to button if not 1 or 2
			self.lbl.grid(row = 1, column = 7)
		else:																#if 1 or 2:
			self.tbl1.delete(*self.tbl1.get_children())						#deletes table
			self.lbl.destroy()												#deletes label if present
			self.entr1.delete(0, 'end')										#clears value from entry

			for i, obj in enumerate(self.K2):								#updating table of selected storms
				self.tbl1.insert('', 'end', text = ' ', values = (self.opt[i], str(obj), str(len(self.idx2[i]))))

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
		#continues to next screen, destroys this window
	def cont(self):
		self.new.destroy()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
		#returns list of number of subclusters to rest of code
	def get_list(self):
		self.K2_ret = self.K2
		return self.K2_ret

#~~~~~~~~~~~~~~~~~~~~~~~~~~
class plotting_interactive():
	#plotting results and selecting storms
	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def __init__(self, rep_storms, idx, K2, idx2, ts, time, bin_sorted, station_id, project_id, et_t, bin_range, outpath, ft_m_lbl):
		#~~~~~~~~~~~~~~~~~~~~~~~~~~
			#setting up variables
		self.rep_storms = rep_storms
		self.rep_storms2 = [i for sublist in self.rep_storms for i in sublist]
		self.idx = idx
		self.K2 = K2
		self.idx2 = idx2
		self.ts = ts
		self.time = time
		self.bin_sorted = bin_sorted
		self.stormlist_sel = [i for i in self.ts.columns]
		self.station_id = station_id
		self.project_id = project_id
		self.et_t = et_t
		self.bin_range = bin_range
		self.outpath = outpath
		self.ft_m_lbl = ft_m_lbl

		self.new = Toplevel()												#new window
		self.frame = Frame(self.new, bd = 1, relief = RAISED)				#creating frame in new window
		self.frame2 = Frame(self.new, bd = 1, relief = RAISED)
		self.frame3 = Frame(self.new, bd = 1, relief = RAISED)
		self.opt = []				#includes 'A', 'B', and 'A & B'			#defining label options based on bins, and one or two groups per bin
		self.opt2 = []				#includes 'A', 'B'
		for i, obj in enumerate(self.K2):
			if obj == 1:
				self.opt.append(str(i+1))
				self.opt2.append(str(i+1))
			elif obj == 2:
				self.opt.append(str(i+1) + '-A')
				self.opt.append(str(i+1) + '-B')
				self.opt.append(str(i+1) + '-A & B')

				self.opt2.append(str(i+1) + '-A')
				self.opt2.append(str(i+1) + '-B')


		#for drop down menu
		self.vari = StringVar(self.frame)
		self.drop_dwn1 = OptionMenu(self.frame, self.vari, *self.opt, command = self.bin_num)		#updates what is plotted
		self.drop_dwn1.grid(row = 1, column = 1)

		#creating initial slider and save button
		self.sldr1 = ttk.Scale(self.frame, from_ = 1, to = 10, orient = HORIZONTAL, length = 500)
		self.sldr1.grid(row = 1, column =2, columnspan = 4)
		self.btn1 = ttk.Button(self.frame, text = 'Save Selected Storm')
		self.btn1.grid(row = 1, column = 6)
		self.btn2 = ttk.Button(self.frame2, text = 'Calculate Relative Probabilities', command = self.calc_prob)
		self.btn2.grid(row = 4, column = 1)

		#creating table of selected storms
		self.tbl1 = ttk.Treeview(self.frame2, columns = ('Cluster', 'Selected Storm'), height = len(self.rep_storms2))
		self.tbl1.heading('#1', text = 'Cluster')
		self.tbl1.heading('Selected Storm', text = 'Selected Storm')
		self.tbl1.column('#1', stretch= YES)
		self.tbl1.column('Selected Storm', stretch = YES)
		self.tbl1['show'] = 'headings'
		self.tbl1.grid(row = 1, column = 1, rowspan = 3)

		rep_storms2 = [i for sublist in self.rep_storms for i in sublist]

		for i, obj in enumerate(rep_storms2):
			self.tbl1.insert('', 'end', text = ' ', values = (self.opt2[i], str(obj)))
			#fix this to have number of storms in cluster. If 1 and 0 (in idx2) count 1's and 0's.

		#defining figure, and setting bckgrnd colour
		self.fig = Figure()
		self.ax = self.fig.add_subplot(111)
		# self.ax.set_facecolor('#c2c2d6')
		self.canvas = FigureCanvasTkAgg(self.fig, master = self.frame3)
		self.canvas.show()
		self.canvas.get_tk_widget().pack(side = 'bottom', fill = 'both', expand = 1)
		self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.frame3)
		self.toolbar.update()
		self.toolbar.pack(side = TOP)

		self.new.grid_columnconfigure(1, weight = 1)
		self.new.grid_columnconfigure(2, weight = 1)
		self.new.grid_columnconfigure(3, weight = 1)
		self.new.grid_columnconfigure(4, weight = 1)
		self.new.grid_columnconfigure(5, weight = 1)

		self.new.grid_rowconfigure(1, weight = 1)
		self.new.grid_rowconfigure(2, weight = 1)
		self.new.grid_rowconfigure(3, weight = 1)
		self.new.grid_rowconfigure(4, weight = 1)
		self.new.grid_rowconfigure(5, weight = 1)

		self.frame.grid(row = 5, column = 1,  columnspan = 4, sticky=E+W)
		self.frame2.grid(row = 1, column = 5, rowspan = 5, sticky=N+S)
		self.frame3.grid(row = 1, column = 1, rowspan = 4, columnspan = 4, sticky=N+S+E+W)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def bin_num(self, value):
		#displays the storms in a given bin, and allows user to choose new storm
		self.ax.cla()											#clearing plot

		if len(value) > 2:										#if there are two groups in a bin (2 sub-bins)
			self.value_1 = int(value.split('-')[0]) - 1			#bin number, indexed to be one less than display value (starts at 0)
			self.value_2a = value.split('-')[1]					#second part of bin number ('A' or 'B')
			
			if self.value_2a == 'A':
				self.value_2 = 0											#for idx2
				self.rep_storm_temp = self.rep_storms[self.value_1][0]		#k-means defined representative storm
				plt_intrctv = True
			elif self.value_2a == 'B':
				self.value_2 = 1											#for idx2
				self.rep_storm_temp = self.rep_storms[self.value_1][1]		#k-means defined representative storm
				plt_intrctv = True

			elif self.value_2a == 'A & B':
				# self.value_2 = 0
				# self.value_2b = 1
				rep_storm_temp_0 = self.rep_storms[self.value_1][0]
				rep_storm_temp_1 = self.rep_storms[self.value_1][1]
				plt_intrctv = False

				a0 = np.argwhere(self.idx==self.bin_sorted[self.value_1])			#index of storms in bin
				b0 = np.argwhere(np.array(self.idx2[self.value_1]) == 0)	#index of storms in sub-bin
				b1 = np.argwhere(np.array(self.idx2[self.value_1]) == 1)

				b = len(b0) + len(b1)

				self.temp_storms = [self.stormlist_sel[ii[0]] for ii in a0]			#storm names of storms in bin
				self.temp_storms_0 = [self.temp_storms[ii[0]] for ii in b0]			#storm names of storms in sub-bin
				self.temp_storms_1 = [self.temp_storms[ii[0]] for ii in b1]

				self.ts_temp_0 = self.ts[self.temp_storms_0]							#elevation time series of storms in sub-bin
				self.ts_temp_1 = self.ts[self.temp_storms_1]
				self.time_temp_0 = self.time[self.temp_storms_0]
				self.time_temp_1 = self.time[self.temp_storms_1]

				if self.sldr1:					
					self.sldr1.destroy()
				if self.btn1:					
					self.btn1.destroy()

				self.ax.plot(self.time_temp_0, self.ts_temp_0, color = '#c994c7', linewidth = 1)
				self.ax.plot(self.time_temp_1, self.ts_temp_1, color = '#9ecae1', linewidth = 1)
				self.ax.plot(self.time_temp_0[rep_storm_temp_0], self.ts_temp_0[rep_storm_temp_0], color = '#dd1c77', linewidth = 3, label = 'Selected: %s' %rep_storm_temp_0)
				self.ax.plot(self.time_temp_1[rep_storm_temp_1], self.ts_temp_1[rep_storm_temp_1], color = '#3182bd', linewidth = 3, label = 'Selected: %s' %rep_storm_temp_1)
				self.ax.set_title('Cluster Number: %s\nNumber of storms in Cluster: %s' %(value, b))
				self.ax.set_xlabel('Time (days)')
				self.ax.set_ylabel('Elevation (%s)' %self.ft_m_lbl)
				self.ax.grid(True)		
				self.ax.legend()
				self.canvas.draw()

		else:																#if there is one group in a bin (1 sub-bin)
			self.value_1 = int(value) - 1									#bin number, indexed to be one less than display value (starts at 0)
			self.value_2a = None											#None, for save storm below
			self.value_2 = self.idx2[self.value_1][0]						#for idx2, set to 0
			self.rep_storm_temp = self.rep_storms[self.value_1][0]				#k-means defined representative storm
			plt_intrctv = True

		if plt_intrctv == True:
			a = np.argwhere(self.idx==self.bin_sorted[self.value_1])			#index of storms in bin
			b = np.argwhere(np.array(self.idx2[self.value_1]) == self.value_2)	#index of storms in sub-bin

			self.temp_storms = [self.stormlist_sel[ii[0]] for ii in a]			#storm names of storms in bin
			self.temp_storms = [self.temp_storms[ii[0]] for ii in b]			#storm names of storms in sub-bin

			self.ts_temp = self.ts[self.temp_storms]							#elevation time series of storms in sub-bin
			self.time_temp = self.time[self.temp_storms]						#time series of storms in sub-bin
			
			
			#clears previous slider, and sets up new slider
			if self.sldr1:					
				self.sldr1.destroy()
			self.sldr1 = Scale(self.frame, from_ = 1, to = len(b), orient = HORIZONTAL, showvalue=1, length = 500,
				label = 'Move Slider to Select Storm', command = self.select_storm)	

			self.sldr1.set(np.argwhere(np.array(self.temp_storms) == np.array(self.rep_storm_temp))[0][0]+1)
			self.sldr1.grid(row = 1, column =2, columnspan = 4)

			#clears 'Save storms' button, and sets up new button
			if self.btn1:					
				self.btn1.destroy()
			self.btn1 = ttk.Button(self.frame, text = 'Save Selected Storm', command = self.save_storm)
			self.btn1.grid(row = 1, column = 6)

			#plots storms in sub-bin, k-means selected storm, and user selected storms
			self.ax.plot(self.time_temp, self.ts_temp, 'k', linewidth = .5)
			self.ax.plot(self.time_temp[self.rep_storm_temp], self.ts_temp[self.rep_storm_temp], 'b', linewidth = 2, label = 'Selected : %s' %self.rep_storm_temp)
			self.line3 = self.ax.plot(self.time_temp[self.rep_storm_temp], self.ts_temp[self.rep_storm_temp], 'r', linewidth = 3, label = 'Viewing : %s' %self.rep_storm_temp)
			self.ax.grid(True)		
			self.ax.legend()
			self.ax.set_title('Cluster Number: %s\nNumber of storms in Cluster: %s' %(value, len(b)))
			self.ax.set_xlabel('Time (days)')
			self.ax.set_ylabel('Elevation (%s)' %self.ft_m_lbl)
			self.canvas.draw()

			self.toolbar.update()												#update toolbar

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def select_storm(self, value):
		#part of slider, changes the user selected storm in plot
		value = int(value) - 1
		self.line3.pop(0).remove()		#removes old 'red' line (in plot)
		self.line3 = self.ax.plot(self.time_temp[self.temp_storms[value]], self.ts_temp[self.temp_storms[value]], 'r', 
			linewidth = 3, label = 'Viewing: %s' %self.temp_storms[value])			#new 'red' line (in plot)
		self.ax.legend()				#creating legend
		self.canvas.draw()				#drawing to canvas

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def save_storm(self):
		#part of button; updates representative storms in list
		#various indexing options
		self.tbl1.delete(*self.tbl1.get_children())

		if self.value_2a == 'A':
			self.rep_storms[self.value_1][0] = self.temp_storms[self.sldr1.get()-1]
		elif self.value_2a == 'B':
			self.rep_storms[self.value_1][1] = self.temp_storms[self.sldr1.get()-1]
		else:
			self.rep_storms[self.value_1] = [self.temp_storms[self.sldr1.get()-1]]
		

		#updating table of selected storms
		rep_storms2 = [i for sublist in self.rep_storms for i in sublist]
		for i, obj in enumerate(rep_storms2):
			self.tbl1.insert('', 'end', text = ' ', values = (self.opt2[i], str(obj)))

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def calc_prob(self):
		self.new.destroy()

		if self.et_t == 'Tropical_Synthetic':
			os.chdir(sys._MEIPASS)
			# os.chdir('G:\\FY17\\automation_practice\\python\\gui\\pyinstaller')
			if self.project_id == 'USACE_NACCS':
				prob_file = 'NACCS_TS_Sim0_Post0_ST_Stat_SRR.h5'
				prob_available = True
			elif self.project_id == 'USACE_Texas':
				prob_file = 'S2G_TS_Sim0_Post0_ST_Stat_SRR_dylan.h5'
				prob_available = True
			else:
				prob_available = False


			if prob_available == True:
				prob_val = []

				for i in self.ts.columns:
					i = i + ' SRP'
					temp = h5reader_prob(prob_file, 'Storm Relative Probabilities - 2', i, int(self.station_id))
					prob_val.append(temp)

				prob = pd.DataFrame(prob_val).T
				prob.columns = self.ts.columns

				#===============================
				# loop to create 'K' bins with stormlist inside
				stormlist_bin = []
				strms_n_bin = []

				count = 0
				for i, obj in enumerate(self.bin_sorted):
					try:
						self.idx2[int(i)]
						temp = np.argwhere(self.idx==obj)
						temp = [ii[0] for ii in temp]

						if max(self.idx2[int(i)]) == 0:
							stormlist_temp = [self.stormlist_sel[ii] for ii in temp]
							stormlist_bin.append(stormlist_temp)
						else:
							stormlist_temp = [self.stormlist_sel[ii] for ii in temp]
							temp1 = np.argwhere(np.array(self.idx2[int(i)]) == 0)
							temp2 = np.argwhere(np.array(self.idx2[int(i)]) == 1)

							temp1 = [ii[0] for ii in temp1]
							temp2 = [ii[0] for ii in temp2]

							storm_temp1 = [stormlist_temp[ii] for ii in temp1]
							storm_temp2 = [stormlist_temp[ii] for ii in temp2]

							stormlist_bin.append(storm_temp1)
							stormlist_bin.append(storm_temp2)

							count += 1

					except IndexError:
						print('+One defined cluster did not contain any storm events')

					#new ==== finds which bin peaks would have fallen in
					# for ii, obj in enumerate(peaks_nis):
					# 	if (obj > self.bin_range[i][0]) and (obj < self.bin_range[i][1]):
					# 		if max(self.idx2[int(i)]) == 0:
					# 			stormlist_bin[i].append(not_in_list[ii])
					# 		else:
					# 			ts_nan = nan_trim(self.ts)
					# 			# temp_nis = self.ts[not_in_list[ii]]
					# 			# temp_storms_1 = self.ts[temp1]
					# 			# temp_storms_2 = self.ts[temp2]
					# 			temp_nis = ts_nan[not_in_list[ii]]
					# 			temp_storms_1 = ts_nan[temp1]
					# 			temp_storms_2 = ts_nan[temp2]

					# 			# diff1 = pd.DataFrame.sum(abs(temp_storms_1 - temp_nis))
					# 			# diff2 = pd.DataFrame.sum(abs(temp_storms_2 - temp_nis))

					# 			diff1 = pd.DataFrame(abs(temp_storms_1.values.transpose() - temp_nis.values.transpose())).T
					# 			diff2 = pd.DataFrame(abs(temp_storms_2.values.transpose() - temp_nis.values.transpose())).T
					# 			# diff1 = temp_storms_1.sub(temp_nis)
								
					# 			diff1 = diff1.values.sum()
					# 			diff2 = diff2.values.sum()

					# 			if diff1 < diff2:
					# 				stormlist_bin[count - 1].append(not_in_list[ii])
					# 			else:
					# 				stormlist_bin[count].append(not_in_list[ii])
					# count += 1

				#===============================
				# loop to determine the representative probability
				prob_sel = [[] for i in range(0,len(stormlist_bin))]
				rep_prob = []
				for i, obj in enumerate(stormlist_bin):
					strms_n_bin.append(len(obj))
					temp1 = stormlist_bin[i]
					for ii, obj2 in enumerate(temp1):
						if prob[obj2].any():
							temp2 = float(prob[obj2])
						else:
							temp2 = 0
						prob_sel[i].append(temp2)

					rep_prob.append(math.fsum(prob_sel[i]))         #floating sum for accuracy 
			# ================================================================
			elif prob_available == False:
				rep_prob = ['No Probabilities Available'] * len(self.bin_sorted)

		elif self.et_t == 'Extratropical_Historical':
			rep_prob = []
			strms_n_bin = []
			for i, obj in enumerate(self.rep_storms):
				# a = np.argwhere(self.idx==self.bin_sorted[i])			#index of storms in bin
				for ii in range(len(obj)):
					b = np.argwhere(np.array(self.idx2[i]) == ii)			#index of storms in sub-bin
					strms_n_bin.append(len(b))
					rep_prob.append(len(b))

		# ================================================================
		#summary/results to new window

		self.results = Toplevel()						#new window
		self.frame = Frame(self.results, bd = 1, relief = RAISED)				#creating frame in new window
		self.frame.grid(column = 1, row = 1, sticky = N+S+E+W)

		self.results.grid_columnconfigure(1, weight = 1)
		self.results.grid_rowconfigure(1, weight = 1)



		self.tbl1 = ttk.Treeview(self.frame, columns = ('Cluster', 'Selected Storm', 'Number of Storms Represented', 'Relative Probability'), height = len(self.rep_storms2))

		self.tbl1.heading('#1', text = 'Cluster')
		self.tbl1.heading('Selected Storm', text = 'Selected Storm')
		self.tbl1.heading('Number of Storms Represented', text = 'Number of Storms Represented')
		self.tbl1.heading('Relative Probability', text = 'Relative Probability')

		self.tbl1.column('#1', stretch= YES)
		self.tbl1.column('Selected Storm', stretch = YES)
		self.tbl1.column('Number of Storms Represented', stretch = YES)
		self.tbl1.column('Relative Probability', stretch = YES)

		self.tbl1['show'] = 'headings'
		self.tbl1.grid(row = 1, column = 1, rowspan = 3)

		rep_storms2 = [i for sublist in self.rep_storms for i in sublist]
		for i, obj in enumerate(rep_storms2):
			self.tbl1.insert('', 'end', text = ' ', values = (self.opt2[i], str(obj), strms_n_bin[i], round(rep_prob[i], 6)))


		# ================================================================
		#summary/results to output file(s)
		os.chdir(self.outpath)
		filename = ('Selection_Summary_%s.csv' %self.et_t.split('_')[0])
		thefile = open(filename, 'w')
		for i, obj in enumerate(rep_storms2):
			if i == 0:
				thefile.write('%s,%s,%s,%s\n' %('Cluster', 'Selected Storm', 'Number of Storms Represented', 'Relative Probability'))

			thefile.write('%s,%s,%s,%s\n' %(self.opt2[i], str(obj), strms_n_bin[i], rep_prob[i]))
		thefile.close()

		filename = ('Selection_Clusters_%s.csv' %self.et_t.split('_')[0])
		thefile = open(filename, 'w')
		for i, obj in enumerate(self.bin_range):
			if i == 0:
				thefile.write('Units:,%s\n' %self.ft_m_lbl)
				thefile.write('%s,%s\n' %('Cluster Lower Values', 'Cluster Upper Values'))
			thefile.write('%s,%s\n' %(self.bin_range[i][0], self.bin_range[i][1]))
		thefile.close()



#~~~~~~~~~~~~~~~~~~~~~~~~~~
class Run_StormSelection():
	#to set up GUI 
	i_path = None
	stormrad = None
	outpath = None
	user_bin = None
	kmeans = False
	smooth = False
	lbl_h5 = None
	lbl_csv = None
	lbl_outpath = None
	lbl_usr_bin = None
	ft_m = False
	ft_m_lbl = 'm.'
	user_bin_ft_m = None
	stormlist = None
	kde_TF = False
	bw_est = None

	sldr1 = None
	sldr1_lbl1 = None
	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def __init__(self, root):

		self.frame1 = Frame(root, bd = 1, relief = RAISED)			#creating four frames in initial window
		self.frame2 = Frame(root, bd = 1, relief = RAISED)
		self.frame3 = Frame(root, bd = 1, relief = RAISED)
		self.frame4 = Frame(root, bd = 1, relief = RAISED)

		self.frame1.grid(row = 1, column = 1, sticky=N+S+E+W)		#frame locations
		self.frame2.grid(row = 2, column = 1, sticky=N+S+E+W)
		self.frame3.grid(row = 3, column = 1, sticky=N+S+E+W)
		self.frame4.grid(row = 4, column = 1, sticky=N+S+E+W)

		root.grid_columnconfigure(1, weight = 1)					#stretches frames to fill windows
		root.grid_rowconfigure(1, weight = 1)
		root.grid_rowconfigure(2, weight = 1)
		root.grid_rowconfigure(3, weight = 1)
		root.grid_rowconfigure(4, weight = 1)

		#selecting .h5 file
		self.btn_i_path = ttk.Button(self.frame1, text = 'Browse Time Series (*.h5)', command=self.Browse_h5)
		self.btn_i_path.grid(column = 1, row = 1, sticky = W, pady = 3)

		#selecting .csv file
		self.btn_stormrad = ttk.Button(self.frame1, text = 'Browse Select Storms (*.csv)', command=self.Browse_csv)
		self.btn_stormrad.grid(column = 1, row = 2, sticky = W, pady = 3)

		#selecting output path
		self.btn_output = ttk.Button(self.frame1, text = 'Location for Output Files', command = self.Browse_outpath)
		self.btn_output.grid(column = 1, row = 3, sticky = W, pady = 3)

		#Entering User_bin_values
		self.lbl = ttk.Label(self.frame2, text = "User Defined Cluster Limits (comma delimited):").grid(column=1, row = 1, sticky = W)
		self.lbl2 = ttk.Label(self.frame2, text = self.ft_m_lbl).grid(column = 3, row = 1, sticky = W)
		self.user_bin_entry = ttk.Entry(self.frame2, width = 51)
		self.user_bin_entry.grid(column=2, row = 1, sticky = W+E, pady = 3)
		self.btn_save = ttk.Button(self.frame2, text = 'Save User Defined Clusters', command = self.save_bin)
		self.btn_save.grid(column = 4, row = 1, sticky = E, pady = 3)

		# self.lbl_or = ttk.Label(self.frame2, text = 'OR').grid(column = 1, row = 2)
		self.bin_btn = ttk.Button(self.frame2, text = 'Browse Cluster Limits (*.csv)', command = self.browse_bin).grid(column = 1, row = 3, sticky = W)


		self.checkCmd_kmeans = IntVar()
		self.k_means_check = Checkbutton(self.frame2, text = 'Algorithm Defined Upper and Lower Cluster Limits', variable = self.checkCmd_kmeans, command = self.T_F_conv_kmeans)
		self.k_means_check.grid(column = 1, row = 4, sticky = W, pady = 3)

		#checkboxes for True/False filter and smoothing
		self.checkCmd_smth = IntVar()
		self.checkCmd_ft_m = IntVar()

		self.btn2 = Checkbutton(self.frame3, text = 'Apply 3pt. Smoothing', variable = self.checkCmd_smth, command = self.T_F_conv_smooth)
		self.btn2.grid(column=1, row=1, sticky = W, pady = 3)

		self.btn_ft_m = Checkbutton(self.frame3, text = 'Perform Analysis in Feet', variable = self.checkCmd_ft_m, command = self.T_F_conv_ft_m)
		self.btn_ft_m.grid(column = 1, row = 2, sticky = W, pady = 3)

		#run and clear input buttons
		self.btn3 = ttk.Button(self.frame4, text = 'Run', command = self.begin_runs).grid(column=1, row = 1, sticky=W, padx = 10)
		self.btn4 = ttk.Button(self.frame4, text = 'Clear Input Values', command = self.clear_input).grid(column = 2, row = 1, sticky = E, padx = 10)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def Browse_h5(self):
		self.filename_h5 = askopenfilename(filetypes=(("h5 files", "*.h5"), ("All files", "*.*") ))		#browse for file
		self.i_path = str(os.path.normpath(self.filename_h5))											#path from above
		self.i_path_name = os.path.split(self.i_path)[-1]												#filename

		if self.lbl_h5:														#gets rid of previous label
			self.lbl_h5.destroy()

		self.lbl_h5 = ttk.Label(self.frame1, text = self.i_path_name)		#"prints" filename 
		self.lbl_h5.grid(column = 2, row = 1, sticky= W)					#location of above

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def Browse_csv(self):
		self.filename_csv = askopenfilename(filetypes =  (("csv files", "*.csv"), ("All files", "*.*") ))
		self.stormrad = str(os.path.normpath(self.filename_csv))
		self.stormrad_name = os.path.split(self.stormrad)[-1]

		self.stormpath = os.path.split(self.stormrad)[0]

		os.chdir(self.stormpath)						#creates list of storms in CSV
		self.stormlist = np.genfromtxt(self.stormrad_name, delimiter = ',', skip_header = 1, usecols = 0, dtype = str)

		if self.lbl_csv:
			self.lbl_csv.destroy()

		self.lbl_csv = ttk.Label(self.frame1, text = self.stormrad_name)
		self.lbl_csv.grid(column = 2, row = 2, sticky= W)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def Browse_outpath(self):
		self.outpath = askdirectory()

		if self.lbl_outpath:
			self.lbl_outpath.destroy()

		self.lbl_outpath = ttk.Label(self.frame1, text = self.outpath)
		self.lbl_outpath.grid(column = 2, row = 3, sticky = W)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def browse_bin(self):
		self.user_bin_file = askopenfilename(filetypes = ( ('csv files', '*.csv'), ('txt files', '*.txt'), ('All files', '*.*') ) )

		os.chdir(os.path.split(self.user_bin_file)[0])
		self.user_bin = np.genfromtxt(os.path.split(self.user_bin_file)[-1], delimiter = ',', skip_header = 2)			#reading bin file
		self.user_bin_ft_m = np.genfromtxt(os.path.split(self.user_bin_file)[-1], delimiter = ',', max_rows = 1, dtype = str)[1]	#units of file

		self.user_bin_entry.delete(0, 'end')

		if self.lbl_usr_bin:
			self.lbl_usr_bin.destroy()
		if self.sldr1_lbl1:
			self.sldr1_lbl1.destroy()
			self.sldr1_lbl2.destroy()
			self.sldr1.destroy()
			self.kde_TF = False
			self.checkCmd_kmeans.set(0)

		self.lbl_usr_bin = ttk.Label(self.frame2, text = self.user_bin)
		self.lbl_usr_bin.grid(column = 2, row = 3, sticky = W)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def save_bin(self):
		if self.user_bin_entry.get():											#reading manually defined cluster limits
			self.temp = self.user_bin_entry.get()
			self.user_bin = self.temp.split(',')								#splitting at ','
			self.user_bin = [float(i) for i in self.user_bin]					#to list

			if sorted(self.user_bin) == self.user_bin:							#check if list is sorted, raises error if not
				self.user_bin = self.user_bin
			else:
				raise ValueError("User defined clusters must be ascending")

			user_bin_min = self.user_bin[:-1]									#lower values of clusters
			user_bin_max = self.user_bin[1:]									#upper values of clusters
			self.user_bin = np.column_stack((np.array(user_bin_min), np.array(user_bin_max)))

		elif not self.user_bin_entry.get():
			self.user_bin = None

		if self.lbl_usr_bin:
			self.lbl_usr_bin.destroy()
		if self.sldr1_lbl1:
			self.sldr1_lbl1.destroy()
			self.sldr1_lbl2.destroy()
			self.sldr1.destroy()
			self.kde_TF = False
			self.checkCmd_kmeans.set(0)

		self.lbl_usr_bin = ttk.Label(self.frame2, text = self.user_bin)
		self.lbl_usr_bin.grid(column = 2, row = 3, sticky = W)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def T_F_conv_kmeans(self):


		if self.checkCmd_kmeans.get():			#conversion of checkbox to True/False

			self.user_bin = None
			self.user_bin_entry.delete(0, 'end')
			if self.lbl_usr_bin:
				self.lbl_usr_bin.destroy()

			self.kde_TF = True
			self.sldr1_lbl1 = ttk.Label(self.frame2, text = 'Less Clusters')
			self.sldr1_lbl1.grid(column = 1, row = 5, sticky = E)
			self.sldr1_lbl2 = ttk.Label(self.frame2, text = 'More Clusters')
			self.sldr1_lbl2.grid(column = 3, row = 5, sticky = W)
			self.sldr1 = Scale(self.frame2, from_ = 1, to = 5, orient = HORIZONTAL, showvalue=0, length = 300,
				command = self.bw_sel)	
			self.sldr1.set(3)
			self.sldr1.grid(column = 2, row = 5)

		else:
			self.sldr1.destroy()
			if self.sldr1_lbl1:
				self.sldr1_lbl1.destroy()
				self.sldr1_lbl2.destroy()
			self.kde_TF = False


	def bw_sel(self, value):
		self.bw_est = self.sldr1.get()


	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def T_F_conv_smooth(self):
		if self.checkCmd_smth.get():			#conversion of checkbox to True/False
			self.smooth = True
		else:
			self.smooth = False

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def T_F_conv_ft_m(self):
		if self.checkCmd_ft_m.get():			#conversion of checkbox to ft./m.
			self.ft_m = True
			self.ft_m_lbl = 'ft.'
		else:
			self.ft_m = False
			self.ft_m_lbl = 'm.'

		if self.lbl2:
			self.lbl2.destroy()
		self.lbl2 = ttk.Label(self.frame2, text = self.ft_m_lbl).grid(column = 3, row = 1, sticky = W)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def begin_runs(self):

		if self.ft_m_lbl != self.user_bin_ft_m:			#conversions from ft. to m. 
			if self.user_bin_ft_m == 'm.':
				self.user_bin = [i/.3048 for i in self.user_bin]
			elif self.user_bin_ft_m == 'ft.':
				self.user_bin = [i*.3048 for i in self.user_bin]

		if not self.i_path:								#check for input path and output path
			raise ValueError('Input path not defined')

		if not self.outpath:
			raise ValueError('Output path not defined')


		bin_check = False								#check for clustering method (manual, previous analysis, or kmeans)
		if self.user_bin is not None:
			bin_check = True
		elif self.kde_TF == True:
			bin_check = True

		if bin_check == False:
			raise ValueError('Select cluster binning method')

		#begining runs
		ts, time_step, station_id, project_id, et_t = loading_data(self.i_path, self.stormlist, self.ft_m)
		ts, time = organizing_data(ts, time_step, smooth = self.smooth)

		time = time/1440								#minutes to days
		bin_sorted, idx, K, bin_range = create_bins(ts, self.user_bin, self.kde_TF, self.bw_est)
		rep_storms, K2, idx2, bin_sorted = k_means(ts, time, bin_sorted, idx, self.ft_m_lbl)

		#plotting results
		self.a = plotting_interactive(rep_storms, idx, K2, idx2, ts, time, bin_sorted, station_id, project_id, et_t, bin_range, self.outpath, self.ft_m_lbl)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	def clear_input(self):
		#clear input values and reset initial window
		self.i_path = None
		self.i_path_name = None
		self.stormlist = None
		self.et_t = None

		if self.lbl_h5:				#destroying label
			self.lbl_h5.destroy()

		self.stormrad = None
		self.stormrad_name = None
		if self.lbl_csv:
			self.lbl_csv.destroy()

		self.outpath = None
		if self.lbl_outpath:
			self.lbl_outpath.destroy()

		self.user_bin = None
		self.user_bin_entry.delete(0, 'end')
		if self.lbl_usr_bin:
			self.lbl_usr_bin.destroy()

		self.kde_TF = False
		self.checkCmd_kmeans.set(0)

		self.smooth = False
		self.checkCmd_smth.set(0)

		self.ft_m = False
		self.ft_m_lbl = 'm.'
		self.user_bin_ft_m = None
		self.checkCmd_ft_m.set(0)

		if self.sldr1_lbl1:
			self.sldr1.destroy()
			self.sldr1_lbl1.destroy()
			self.sldr1_lbl2.destroy()

		self.kde_TF = False
		self.bw_est = None


#~~~~~~~~~~~~~~~~~~~~~~~~~~
#running all of the above
root = Tk()												#create main window
root.title('Representative Storm Selection Tool')		#name of window
root.geometry("900x700")								#size of window

prompt = Run_StormSelection(root)						#initiating run
root.mainloop()
#~~~~~~~~~~~~~~~~~~~~~~~~~~


