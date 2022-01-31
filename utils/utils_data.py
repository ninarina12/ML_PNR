import numpy as np
import pandas as pd
import os
import json

from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.utils_simulate import u

# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'


def parse_metadata(dir_name):
	# extract relevant parameters from metadata file
	with open(dir_name + '/metadata.txt') as f:
		meta = json.load(f)
		layers = meta['PARAMETERS']['layers']
		rho = meta['PARAMETERS']['densities (g/cm^3)']
		M = meta['PARAMETERS']['molar masses (g/mol)']
		N = meta['PARAMETERS']['number of q-points']
		q_min = meta['PARAMETERS']['q_min (A^-1)']
		q_max = meta['PARAMETERS']['q_max (A^-1)']
	return layers, rho, M, N, q_min, q_max


def parse_labels(df, exclude_s=False, exclude_inst=False, add_mt=False, custom=None):
	# parse and format labels for plotting selected parameters
	
	# drop columns with duplicate values (e.g. dens columns if multiple sublayers)
	cols = [k for k in df.columns if '.' in k]
	df = df.drop(columns=cols)

	if add_mt:
		if 'd_iAFM' in df.columns: df['mt_iAFM'] = df['d_iAFM']*df['magn_iAFM']
		df['mt_prox'] = df['d_prox']*df['magn_prox']

	# parse y-values
	y_data = df.values
	y_columns = list(df.columns)
	
	if custom:
		y_header = [l for (k,l) in enumerate(df.columns) if k in custom]
	
	else:
		# exclude columns with constant values (e.g. substrate density)
		y_header = df.std()[(df.std() > 1e-15)].index.tolist()

		if exclude_s: y_header = [l for l in y_header if not l.startswith('s_')]
		if exclude_inst:
			y_header.remove('I_bkg')
			y_header.remove('I_res')
		
	y_ids = np.array([list(df.columns).index(l) for l in y_header])

	# translate to symbols
	y_labels = [r'$' + l.split('_')[0] + '_{' + l.split('_')[1] + '}$' for l in y_header]
	sym_old = ['dens_', 'd_', 's_', 'magn_']
	sym_new = [r'\rho_', 't_', r'\sigma_', 'm_']
	for old, new in zip(sym_old, sym_new):
		y_labels = [l.replace(old, new) for l in y_labels]

	# GenX units
	y_units = ['(f.u./'+r'$nm^3$)']*len([l for l in y_header if l.startswith('dens_')]) + \
			  ['(nm)']*len([l for l in y_header if l.startswith('d_')]) + \
			  ['(nm)']*len([l for l in y_header if l.startswith('s_')]) + \
			  [r'($\mu_B$'+'/f.u.)']*len([l for l in y_header if l.startswith('magn_')]) + \
			  ['']*len([l for l in y_header if l.startswith('I_')])

	# conventional units
	y_units_ = ['(g/'+r'$cm^3$)']*len([l for l in y_header if l.startswith('dens_')]) + \
			   ['(nm)']*len([l for l in y_header if l.startswith('d_')]) + \
			   ['(nm)']*len([l for l in y_header if l.startswith('s_')]) + \
			   ['(emu/'+r'$cm^3$)']*len([l for l in y_header if l.startswith('magn_')]) + \
			   ['']*len([l for l in y_header if l.startswith('I_')])

	if add_mt:
		if 'd_iAFM' in df.columns:
			y_units += [r'($\mu_B \cdot$'+'nm/f.u.)']
			y_units_ += ['(emu '+r'$\cdot nm/$'+r'$cm^3$)']
		y_units += [r'($\mu_B \cdot$'+'nm/f.u.)']
		y_units_ += ['(emu '+r'$\cdot nm/$'+r'$cm^3$)']

	return y_data, y_columns, y_header, y_ids, y_labels, y_units, y_units_


def drop_duplicates_list(l):
	s = set()
	s_add = s.add
	return [x for x in l if not (x in s or s_add(x))]


def convert_units(df, M, y_header, y_columns=None):
	# get molar masses
	m = [M[0]] + drop_duplicates_list(M[1:])

	# convert magnetization units to emu/cm^3 and density units to g/cm^3
	if ('y_true' in df.columns) and (y_columns != None):
		df['y_true_'] = df['y_true'].map(lambda x: convert_magn(x, y_columns))
		df['y_true_'] = df['y_true_'].map(lambda x: convert_dens(x, m))
	
	df['y_pred_'] = df['y_pred'].map(lambda x: convert_magn(x, y_header))
	df['y_pred_'] = df['y_pred_'].map(lambda x: convert_dens(x, m))
	return df

	
def convert_dens(x, m):
	u = 1.66054
	f = np.ones(len(x))
	f[:len(m)] = [u*k/1e3 for k in m]
	return x*f


def convert_magn(x, y_header):
	idx_m = [i for i, k in enumerate(y_header) if ('magn' in k) or ('mt' in k)]
	f = np.ones(len(x))
	for k in idx_m:
		if 'prox' in y_header[k]:
			f[k] *= x[y_header.index('dens_TI')]*9.274
		elif 'AFM' in y_header[k]:
			f[k] *= x[y_header.index('dens_AFM')]*9.274
		elif 'FM' in y_header[k]:
			f[k] *= x[y_header.index('dens_FM')]*9.274
		else:
			f[k] = 1.
	return f*x


def read_exp(filename):
	return pd.read_csv(filename, usecols=[0,1,2,3], names=['Q','R','dR','dQ'], skiprows=24, sep='\t', dtype=float)


def get_exp_names(sample_name):
	exp_names = next(os.walk('experiments/' + sample_name))[1]
	exp_names = [k for k in exp_names if '-' not in k]
	exp_names = [exp_names[j] for j in np.argsort([int(i[:i.index('K')]) for i in exp_names])]
	return exp_names


def process_data(data_dir, sample_name, q, seed=12, delta=1., exclude_s=False, exclude_inst=False, add_mt=False, custom=None):
	# read and process simulation data
	xdata_uu =  data_dir + '/xdata_uu.txt'
	xdata_dd =  data_dir + '/xdata_dd.txt'
	ydata =  data_dir + '/ydata.txt'

	x_uu = pd.read_csv(xdata_uu, header=None, sep=' ', dtype=float).values
	x_dd = pd.read_csv(xdata_dd, header=None, sep=' ', dtype=float).values
	x_orig = np.stack([x_uu, x_dd], axis=2)

	# read labels
	y_df = pd.read_csv(ydata, dtype=float)
	x_bkg = np.expand_dims(y_df['I_bkg'].values, 1)

	# add noise
	x_data = add_noise(q, x_orig, x_bkg, seed, delta)

	# parse labels
	y_data, y_columns, y_header, y_ids, y_labels, y_units, y_units_ = parse_labels(
		y_df, exclude_s=exclude_s, exclude_inst=exclude_inst, add_mt=add_mt, custom=custom)

	return x_data, x_orig, y_data, y_columns, y_header, y_ids, y_labels, y_units, y_units_


def process_exp(sample_name, exp_names, q, x_moms=(0,1), norm=True):
	# read and normalize experimental data
	x_exp_list = []
	dx_exp_list = []

	for i in range(len(exp_names)):
		xdata_uu = 'experiments/' + sample_name + '/' + exp_names[i] + '/x_uu.dat'
		xdata_dd = 'experiments/' + sample_name + '/' + exp_names[i] + '/x_dd.dat'
		df_uu = read_exp(xdata_uu)
		df_dd = read_exp(xdata_dd)

		# resample to q points
		x_uu = interp1d(df_uu['Q'].values, df_uu['R'].values, fill_value='extrapolate')(q/10.)
		x_dd = interp1d(df_dd['Q'].values, df_dd['R'].values, fill_value='extrapolate')(q/10.)

		dx_uu = interp1d(df_uu['Q'].values, df_uu['dR'].values, fill_value='extrapolate')(q/10.)
		dx_dd = interp1d(df_dd['Q'].values, df_dd['dR'].values, fill_value='extrapolate')(q/10.)

		x_uu = np.expand_dims(x_uu, axis=0)
		x_dd = np.expand_dims(x_dd, axis=0)
		x_exp = np.stack([x_uu, x_dd], axis=2)

		dx_uu = np.expand_dims(dx_uu, axis=0)
		dx_dd = np.expand_dims(dx_dd, axis=0)
		dx_exp = np.stack([dx_uu, dx_dd], axis=2)

		# normalize
		if norm:
			dx_exp = normalize_log_err(dx_exp, x_exp, x_moms)
			x_exp, _ = normalize_log(x_exp, x_moms)

		x_exp_list += [x_exp]
		dx_exp_list += [dx_exp]

	x_exp = np.concatenate(x_exp_list, axis=0)
	dx_exp = np.concatenate(dx_exp_list, axis=0)

	return x_exp, dx_exp


def add_noise(q, x, x_bkg, seed, delta=1.):
	np.random.seed(seed)

	# approximate error with poisson statistics
	dx_uu = delta*np.sqrt(x_bkg/x[:,:,0])
	dx_dd = delta*np.sqrt(x_bkg/x[:,:,1])
	sig = np.stack([dx_uu, dx_dd], axis=2)
	dx = sig*np.random.standard_normal(size=sig.shape)

	# add noise to simulated data
	x_ = x*(1. + dx)
	x_[x_ <= 0] = x[x_ <= 0]

	# smooth sharp wiggles to match exp. quality
	x_[:,:,0] = x[:,:,0] + smooth(q, x_[:,:,0] - x[:,:,0])
	x_[:,:,1] = x[:,:,1] + smooth(q, x_[:,:,1] - x[:,:,1])
	x_[x_ <= 0] = x[x_ <= 0]

	return x_


def smooth(x, y, w=0):
	# invert
	xi = np.linspace(1./x[-1], 1./x[0], len(x))
	yi = interp1d(1./x, y, kind='linear', fill_value='extrapolate')(xi)
	
	if w:
		# round up window length to nearest odd integer
		w = w + 1 - w%2

	# perform a moving average
		s = np.r_['1', yi[:,w//2+1:0:-1], yi, yi[:,-2:-w//2-1:-1]]
		y_cs = np.cumsum(s, axis=-1)
		yi = (y_cs[:,w:] - y_cs[:,:-w])/w
	
	return interp1d(1./xi, yi, kind='linear', fill_value='extrapolate')(x)


def normalize_log(x, x_moms=None):
	z = np.log10(x)
	if x_moms == None:
		zmin = z.ravel().min()
		zmax = z.ravel().max()
	else:
		zmin = x_moms[0]
		zmax = x_moms[1]
	z = (z - zmin)/(zmax - zmin)
	return z, (zmin, zmax)


def normalize_log_inverse(z, x_moms):
	zmin = x_moms[0]
	zmax = x_moms[1]
	z = z*(zmax - zmin) + zmin
	x = 10.**z
	return x


def normalize_log_err(dx, x, x_moms):
	dz = dx/x
	zmin = x_moms[0]
	zmax = x_moms[1]
	dz = dz/(zmax - zmin)
	return dz


def split_data(x_data, seed=12):
	idx_train, idx_dev = train_test_split(range(len(x_data)), test_size=0.3, random_state=seed)
	idx_valid, idx_test = train_test_split(idx_dev, test_size=2./3., random_state=seed)
	return idx_train, idx_valid, idx_test