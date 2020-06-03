import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

# define constants
e_min = 0
e_max = 6000
e_bin_size = 20
e_num_bins = int((e_max - e_min) / e_bin_size)
e_bin_centers = np.linspace(e_min + e_bin_size / 2, e_max - e_bin_size / 2, e_num_bins)
e_bin_edges = np.linspace(e_min, e_max, e_num_bins + 1)

re_min = -50.0
re_max = 50.0
re_bin_size = 2
re_num_bins = int((re_max - re_min) / re_bin_size)
re_bin_edges = np.linspace(re_min, re_max, re_num_bins + 1)

lw = 0.7
ms = 3
fs = 10
color1 = 'darkolivegreen'
color2 = 'crimson'
color_hist = 'midnightblue'

# format runtime printing
def print_time(time_param):
	if time_param < 60.0:
		if '%.2f' % time_param == '1.00':
			return '%.2f second' % time_param
		return '%.2f seconds' % time_param

	time_param /= 60.0
	if time_param < 60.0:
		if '%.2f' % time_param == '1.00':
			return '%.2f minute' % time_param
		return '%.2f minutes' % time_param

	time_param /= 60.0
	if '%.2f' % time_param == '1.00':
		return '%.2f hour' % time_param
	return '%.2f hours' % time_param

# calculate relative error and restrict to plus or minus 50%
def get_re(true, predict):
	re = []
	for i in range(len(true)):
		if true[i] == 0.0:
			if predict[i] == 0.0: re.append(0.0)
			elif predict[i] < true[i]: re.append(re_min)
			elif predict[i] > true[i]: re.append(re_max)
		else:
			temp = (predict[i] - true[i]) / true[i] * 100
			if temp > re_max: temp = re_max
			elif temp < re_min: temp = re_min
			re.append(temp)
	return re

# plot 1D input, 1D output
def plot_1D(as_hist, model_params, x, ax1_y1, ax1_l1, ax1_y2, ax1_yerr, ax1_l2, ax1_title, ax1_ylabel, ax2_y, ax2_xlabel, ax2_ylabel, dir):
	fig = plt.figure(constrained_layout = True)
	gs = fig.add_gridspec(5, 1)
	ax1 = fig.add_subplot(gs[0:4, :])
	ax2 = fig.add_subplot(gs[-1, :])

	for i in range(len(model_params)):
		ax1.plot([], [], ' ', label = model_params[i])

	if as_hist:
		n1, b, p = ax1.hist(x, bins = e_bin_edges, weights = ax1_y1, label = ax1_l1, histtype = 'step', stacked = True, fill = False, linewidth = lw, color = color1)
		n2, b, p = ax1.hist(x, bins = e_bin_edges, weights = ax1_y2, label = ax1_l2, histtype = 'step', stacked = True, fill = False, linewidth = lw, color = color2)
	else:
		ax1.plot(x, ax1_y1, '.', label = ax1_l1, markersize = ms, color = color1)
		ax1.errorbar(x, ax1_y2, yerr = ax1_yerr, fmt = '.', label = ax1_l2, linewidth = lw, markersize = ms, color = color2)

	ax1.set_xlim(e_min - 5, e_max + 5)
	ax1.legend(loc = 'best', fontsize = fs)
	ax1.set_title(ax1_title)
	ax1.set_ylabel(ax1_ylabel)

	if as_hist:
		re = get_re(n1, n2)
		ax2.hist(e_bin_centers, bins = e_bin_edges, weights = re, linewidth = lw, color = color1)
	else:
		ax2.plot(x, ax2_y, '.', markersize = ms, color = color1)
	ax2.plot(np.linspace(e_min - 5, e_max + 5, 2), [0] * 2, linewidth = lw, markersize = ms, color = color2)

	ax2.set_xlim(e_min - 5, e_max + 5)
	ax2.set_ylim(re_min - 1, re_max + 1)
	ax2.set_xlabel(ax2_xlabel)
	ax2.set_ylabel(ax2_ylabel)

	fig.savefig(dir)
	plt.close(fig)

	if as_hist:
		return re

# plot relative error
def plot_re(fn, re, model_params, title, xlabel, dir):
	plt.figure(fn)

	plt.hist(re, re_bin_edges, color = color_hist)
	for i in range(len(model_params)):
		plt.plot([], [], ' ', label = model_params[i])

	plt.xlim(re_min - 1, re_max + 1)
	plt.legend(loc = 'best', fontsize = fs)
	plt.title(title)
	plt.xlabel(xlabel)

	plt.savefig(dir)
	plt.close(fn)
