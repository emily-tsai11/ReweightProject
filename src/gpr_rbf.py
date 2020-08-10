from time import time
import numpy as np
from mod import print_time, get_re, plot_1D, plot_re
import sklearn.gaussian_process as gp
from sklearn.metrics.pairwise import rbf_kernel
import json, os, ROOT, argparse

# define constants
m_index = 0
pt_index = 1
pz_index = 2

start = time()

# define parser
parser = argparse.ArgumentParser(description = 'ML-Based Reweight Factor Generation: GPR with RBF')
parser.add_argument('--config', required = True, type = str, dest = 'config', metavar = '<config.py>', help = 'config file defining files and variables')

# parse the arguments
args = parser.parse_args()
config = json.loads(open(args.config).read())

# load config file variables
print('loading config file variables & data files...')
start_load = time()

# with open(config['FILES']['events'], 'r') as f:
# 	events = json.load(f)

# i hope this works
config = json.load(open(config['FILES']['events']))

num_train = config['NUM']['train']
num_test = config['NUM']['test']

couplings = config['COUPLINGS']

kernel_l = config['MODEL_PARAM']['kernel_l']
nro = config['MODEL_PARAM']['nro']
a = config['MODEL_PARAM']['a']

print('loading time: ' + print_time(time() - start_load))

# make sure the number of events you want don't exceed the total number of events
if num_train + num_test > len(events):
	print('the size of the training and test sets you input is greater than the number of data points available. please try again.')
	exit()

ckeys = couplings.keys()
for ckey in ckeys:
	print('----------------------------------------------------------------------')

	# define input and output coupling
	input_coupling = 'M%2dK%03d' % (couplings[ckey]['Mi'], couplings[ckey]['Ki'])
	output_coupling = 'M%2dK%03d' % (couplings[ckey]['Mf'], couplings[ckey]['Kf'])

	# define save directory
	save_dir = '../plots/GPR_RBF/' + input_coupling + 'to' + output_coupling + '_ntr' + str(num_train) + 'te' + str(num_test) + '_k' + str(kernel_l) + '_nro' + str(nro) + '_a' + str(a) + '/'

	# create save directory, unless it already exists
	if not os.path.isdir(save_dir):
		os.system('mkdir ' + save_dir)

	# open runtime file
	f = open(save_dir + 'runtimes.txt', 'w')

	# print input and output couplings
	print('coupling ' + ckey + ' of ' + str(len(ckeys)) + ':')
	print('\tinput coupling of: ' + input_coupling)
	print('\toutput coupling of: ' + output_coupling)

	f.write('coupling:\n')
	f.write('\tinput coupling of: ' + input_coupling + '\n')
	f.write('\toutput coupling of: ' + output_coupling + '\n')

	# creating training and testing datasets
	print('creating')
	print('\ttraining dataset with ' + str(num_train) + ' events')
	print('\ttesting dataset with ' + str(num_test) + ' events')

	f.write('creating\n')
	f.write('\ttraining dataset with ' + str(num_train) + ' events\n')
	f.write('\ttesting dataset with ' + str(num_test) + ' events\n')

	ev_num_train = []
	X_train = []
	y_train = []
	ev_num_test = []
	X_test = []
	y_test = []

	for i in range(num_train + num_test):
		e = events[str(i)]
		Top = e['2']
		_4vector = ROOT.TLorentzVector()
		_4vector.SetPxPyPzE(Top['px'], Top['py'], Top['pz'], Top['e'])

		if i < num_train:
			ev_num_train.append(i)
			X_train.append([Top['m'], _4vector.Pt(), Top['pz']])
			y_train.append([e['wts'][output_coupling] / e['wts'][input_coupling]])
		else:
			ev_num_test.append(i)
			X_test.append([Top['m'], _4vector.Pt(), Top['pz']])
			y_test.append([e['wts'][output_coupling] / e['wts'][input_coupling]])

	# get individual parameter inputs
	X_train_m = [X_train[i][m_index] for i in range(len(X_train))]
	X_test_m = [X_test[i][m_index] for i in range(len(X_test))]

	X_train_pt = [X_train[i][pt_index] for i in range(len(X_train))]
	X_test_pt = [X_test[i][pt_index] for i in range(len(X_test))]

	X_train_pz = [X_train[i][pz_index] for i in range(len(X_train))]
	X_test_pz = [X_test[i][pz_index] for i in range(len(X_test))]

	# print(np.reshape(X_train_m, (len(X_train_m), 1)))
	# print(np.reshape(y_train, (len(y_train), 1)))

	# defining model (USED TO HAVE CONSTANT KERNEL)
	kernel1 = gp.kernels.RBF(kernel_l, (1e-3, 1e3)) * gp.kernels.RBF(kernel_l, (1e-3, 1e3)) * gp.kernels.RBF(kernel_l, (1e-3, 1e3))
	kernel2 = rbf_kernel(np.reshape(X_train_m, (len(X_train_m), 1)), np.reshape(y_train, (len(y_train), 1)))
	# print(np.array(kernel1.__call__(np.reshape(X_train_m, (len(X_train_m), 1)), np.reshape(y_train, (len(y_train), 1)))))
	# print(kernel2)
	model = gp.GaussianProcessRegressor(kernel = kernel1, n_restarts_optimizer = nro, alpha = a, normalize_y = False)

	print('kernel has initial guess of: ' + str(kernel_l))
	print('model has n_restarts_optimizer of: ' + str(nro))
	print('model has an alpha of: ' + str(a))

	f.write('kernel has initial guess of: ' + str(kernel_l) + '\n')
	f.write('model has n_restarts_optimizer of: ' + str(nro) + '\n')
	f.write('model has an alpha of: ' + str(a) + '\n')

	# training model
	print('training model...')
	start_train = time()

	model.fit(X_train, y_train)

	print('model parameters: ' + str(model.kernel_))
	print('training time: ' + print_time(time() - start_train))

	f.write('model parameters: ' + str(model.kernel_) + '\n')
	f.write('training time: ' + print_time(time() - start_train) + '\n')

	# testing model on training data
	print('testing model on training data...')
	start_train_fit = time()

	y_predict_train, std_train = model.predict(X_train, return_std = True)

	print('testing on training data time: ' + print_time(time() - start_train_fit))
	f.write('testing on training data time: ' + print_time(time() - start_train_fit) + '\n')

	# testing model on testing data
	print('testing model on testing data...')
	start_test_fit = time()

	y_predict_test, std_test = model.predict(X_test, return_std = True)

	print('testing on testing data time: ' + print_time(time() - start_test_fit))
	f.write('testing on testing data time: ' + print_time(time() - start_test_fit) + '\n')

	# reshape weight arrays for plotting
	y_train = np.array(y_train).reshape(1, len(y_train))[0]
	y_test = np.array(y_test).reshape(1, len(y_test))[0]
	y_predict_train = np.array(y_predict_train).reshape(1, len(y_predict_train))[0]
	y_predict_test = np.array(y_predict_test).reshape(1, len(y_predict_test))[0]

	# calculating relative error on weight predictions and restricting to plus or minus 50%
	re_train = get_re(y_train, y_predict_train)
	re_test = get_re(y_test, y_predict_test)

	# saving training/testing raw data
	print('saving raw data...')
	start_save = time()

	np.savetxt(save_dir + 'train.csv', np.array([ev_num_train, X_train_m, X_train_pt, X_train_pz, y_train, y_predict_train, std_train, re_train]), delimiter = ',')
	np.savetxt(save_dir + 'test.csv', np.array([ev_num_test, X_test_m, X_test_pt, X_test_pz, y_test, y_predict_test, std_test, re_test]), delimiter = ',')

	print('saving raw data time: ' + print_time(time() - start_save))
	f.write('saving raw data time: ' + print_time(time() - start_save) + '\n')

	# creating and saving plots
	print('creating and saving plots...')
	start_plot = time()

	fn = 0
	model_params = ['kernel initial guess: ' + str(kernel_l), 'n_restarts_optimizer: ' + str(nro), 'alpha: ' + str(a), str(model.kernel_)]

	#--------------------------------------------------------------------------------#

	# training weights relative error
	plot_re(
		fn,
		re_train,
		model_params,
		input_coupling + '-->' + output_coupling + ' | rel err, training weights (' + str(num_train) + ')\n(y_predict_train - y_train) / y_train * 100 | restricted to $\pm$50%',
		'relative error (%)',
		save_dir + 're_train.png')
	fn += 1

	# testing weights relative error
	plot_re(
		fn,
		re_test,
		model_params,
		input_coupling + '-->' + output_coupling + ' | rel err, testing weights (' + str(num_test) + ')\n(y_predict_test - y_test) / y_test * 100 | restricted to $\pm$50%',
		'relative error (%)',
		save_dir + 're_test.png')
	fn += 1

	#--------------------------------------------------------------------------------#

	# mass vs. weight training, mass distribution training prediction
	plot_1D(
		False,
		model_params,
		X_train_m, y_train,
			'training dataset',
		y_predict_train, std_train,
			'GPR training prediction',
			input_coupling + '-->' + output_coupling + ' | $M_T$ GPR training prediction (' + str(num_train) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_train,
			'invariant $M_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'm_predict_train.png')

	plot_1D(
		False,
		model_params,
		X_train_m, y_train,
			'training dataset',
		y_predict_train, [0] * len(y_predict_train),
			'GPR training prediction',
			input_coupling + '-->' + output_coupling + ' | $M_T$ GPR training prediction (' + str(num_train) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_train,
			'invariant $M_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'm_predict_train_nostd.png')

	plot_1D(
		True,
		model_params,
		X_train_m, y_train,
			'$M_T$ distribution training',
		y_predict_train, [0] * len(y_predict_train),
			'$M_T$ distribution training prediction',
			input_coupling + '-->' + output_coupling + ' | $M_T$ distribution training prediction (' + str(num_train) + ')',
			' ',
		[],
			'invariant $M_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'm_distribution_train.png')

	# mass vs. weight testing, mass distribution testing prediction
	plot_1D(
		False,
		model_params,
		X_test_m, y_test,
			'testing dataset',
		y_predict_test, std_test,
			'GPR testing prediction',
			input_coupling + '-->' + output_coupling + ' | $M_T$ GPR testing prediction (' + str(num_test) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_test,
			'invariant $M_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'm_predict_test.png')

	plot_1D(
		False,
		model_params,
		X_test_m, y_test,
			'testing dataset',
		y_predict_test, [0] * len(y_predict_test),
			'GPR testing prediction',
			input_coupling + '-->' + output_coupling + ' | $M_T$ GPR testing prediction (' + str(num_test) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_test,
			'invariant $M_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'm_predict_test_nostd.png')

	plot_1D(
		True,
		model_params,
		X_test_m, y_test,
			'$M_T$ distribution testing',
		y_predict_test, [0] * len(y_predict_test),
			'$M_T$ distribution testing prediction',
			input_coupling + '-->' + output_coupling + ' | $M_T$ distribution testing prediction (' + str(num_test) + ')',
			' ',
		[],
			'invariant $M_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'm_distribution_test.png')

	#--------------------------------------------------------------------------------#

	# pt vs. weight training, pt distribution training prediction
	plot_1D(
		False,
		model_params,
		X_train_pt, y_train,
			'training dataset',
		y_predict_train, std_train,
			'GPR training prediction',
			input_coupling + '-->' + output_coupling + ' | $p_T$ GPR training prediction (' + str(num_train) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_train,
			'$p_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pt_predict_train.png')

	plot_1D(
		False,
		model_params,
		X_train_pt, y_train,
			'training dataset',
		y_predict_train, [0] * len(y_predict_train),
			'GPR training prediction',
			input_coupling + '-->' + output_coupling + ' | $p_T$ GPR training prediction (' + str(num_train) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_train,
			'$p_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pt_predict_train_nostd.png')

	plot_1D(
		True,
		model_params,
		X_train_pt, y_train,
			'$p_T$ distribution training',
		y_predict_train, [0] * len(y_predict_train),
			'$p_T$ distribution training prediction',
			input_coupling + '-->' + output_coupling + ' | $p_T$ distribution training prediction (' + str(num_train) + ')',
			' ',
		[],
			'$p_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pt_distribution_train.png')

	# pt vs. weight testing, pt distribution testing prediction
	plot_1D(
		False,
		model_params,
		X_test_pt, y_test,
			'testing dataset',
		y_predict_test, std_test,
			'GPR testing prediction',
			input_coupling + '-->' + output_coupling + ' | $p_T$ GPR testing prediction (' + str(num_test) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_test,
			'$p_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pt_predict_test.png')

	plot_1D(
		False,
		model_params,
		X_test_pt, y_test,
			'testing dataset',
		y_predict_test, [0] * len(y_predict_test),
			'GPR testing prediction',
			input_coupling + '-->' + output_coupling + ' | $p_T$ GPR testing prediction (' + str(num_test) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_test,
			'$p_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pt_predict_test_nostd.png')

	plot_1D(
		True,
		model_params,
		X_test_pt, y_test,
			'$p_T$ distribution testing',
		y_predict_test, [0] * len(y_predict_test),
			'$p_T$ distribution testing prediction',
			input_coupling + '-->' + output_coupling + ' | $p_T$ distribution testing prediction (' + str(num_test) + ')',
			' ',
		[],
			'$p_T$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pt_distribution_test.png')

	#--------------------------------------------------------------------------------#

	# pz vs. weight training, pz distribution training prediction
	plot_1D(
		False,
		model_params,
		X_train_pz, y_train,
			'training dataset',
		y_predict_train, std_train,
			'GPR training prediction',
			input_coupling + '-->' + output_coupling + ' | $p_z$ GPR training prediction (' + str(num_train) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_train,
			'$p_z$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pz_predict_train.png')

	plot_1D(
		False,
		model_params,
		X_train_pz, y_train,
			'training dataset',
		y_predict_train, [0] * len(y_predict_train),
			'GPR training prediction',
			input_coupling + '-->' + output_coupling + ' | $p_z$ GPR training prediction (' + str(num_train) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_train,
			'$p_z$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pz_predict_train_nostd.png')

	plot_1D(
		True,
		model_params,
		X_train_pz, y_train,
			'$p_z$ distribution training',
		y_predict_train, [0] * len(y_predict_train),
			'$p_z$ distribution training prediction',
			input_coupling + '-->' + output_coupling + ' | $p_z$ distribution training prediction (' + str(num_train) + ')',
			' ',
		[],
			'$p_z$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pz_distribution_train.png')

	# pz vs. weight testing, pz distribution testing prediction
	plot_1D(
		False,
		model_params,
		X_test_pz, y_test,
			'testing dataset (' + str(num_test) + ')',
		y_predict_test, std_test,
			'GPR testing prediction',
			input_coupling + '-->' + output_coupling + ' | $p_z$ GPR testing prediction (' + str(num_test) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_test,
			'$p_z$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pz_predict_test.png')

	plot_1D(
		False,
		model_params,
		X_test_pz, y_test,
			'testing dataset (' + str(num_test) + ')',
		y_predict_test, [0] * len(y_predict_test),
			'GPR testing prediction',
			input_coupling + '-->' + output_coupling + ' | $p_z$ GPR testing prediction (' + str(num_test) + ')',
			output_coupling + ' weights / ' + input_coupling + ' weights',
		re_test,
			'$p_z$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pz_predict_test_nostd.png')

	plot_1D(
		True,
		model_params,
		X_test_pz, y_test,
			'$p_z$ (' + str(num_train) + ')',
		y_predict_test, [0] * len(y_predict_test),
			'$p_z$ testing prediction',
			input_coupling + '-->' + output_coupling + ' | $p_z$ distribution testing prediction (' + str(num_test) + ')',
			' ',
		[],
			'$p_z$ (GeV)',
			'relative\nerror (%)',
		save_dir + 'pz_distribution_test.png')

	#--------------------------------------------------------------------------------#

	print('plotting time: ' + print_time(time() - start_plot))
	f.write('plotting time: ' + print_time(time() - start_plot) + '\n')

	# closing file
	f.close()

print('----------------------------------------------------------------------')
print('TOTAL RUNTIME: ' + print_time(time() - start))
