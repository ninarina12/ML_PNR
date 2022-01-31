import numpy as np
import os, sys
import json
import re
import time

from _ctypes import PyObj_FromPtr
from mpi4py import MPI


def blockPrint():
	sys.stdout = open(os.devnull, 'w')


def enablePrint(out=None):
	if out: sys.stdout = out
	else: sys.stdout = sys.__stdout__


class NoIndent(object):
	def __init__(self, value):
		self.value = value


class CustomEncoder(json.JSONEncoder):
	FORMAT_SPEC = '@@{}@@'
	regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

	def __init__(self, **kwargs):
		self.__sort_keys = kwargs.get('sort_keys', None)
		super(CustomEncoder, self).__init__(**kwargs)

	def default(self, obj):
		return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
				else super(CustomEncoder, self).default(obj))

	def encode(self, obj):
		format_spec = self.FORMAT_SPEC
		json_repr = super(CustomEncoder, self).encode(obj)

		for match in self.regex.finditer(json_repr):
			id = int(match.group(1))
			no_indent = PyObj_FromPtr(id)
			json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)
			json_repr = json_repr.replace('"{}"'.format(format_spec.format(id)), json_obj_repr)

		return json_repr


def format_parameters(b_sub, b_main1, b_main2, b_cap,
					  dens_sub, dens_main1, dens_main2, dens_cap,
					  ddens_sub, ddens_main1, ddens_main2,
					  ddens_cap, v_sub, v_main1, v_main2, v_cap,
					  dv_sub, dv_main1, dv_main2, dv_cap):	
	
	# assemble parameters
	b = [b_sub, b_main1, b_main2, b_cap]
	dens = [dens_sub, dens_main1, dens_main2, dens_cap]
	ddens = [ddens_sub, ddens_main1, ddens_main2, ddens_cap]
	v = [v_sub, v_main1, v_main2, v_cap]
	dv = [dv_sub, dv_main1, dv_main2, dv_cap]

	# replicate properties for subdivided layers
	dens_main1_rep = np.repeat(dens_main1, v_main1.shape[0], axis=0)
	ddens_main1_rep = np.repeat(ddens_main1, dv_main1.shape[0], axis=0)
	dens_main2_rep = np.repeat(dens_main2, v_main2.shape[0], axis=0)
	ddens_main2_rep = np.repeat(ddens_main2, dv_main2.shape[0], axis=0)

	# assemble all parameters
	sub = np.concatenate([dens_sub, v_sub], axis=1)
	dsub = np.concatenate([ddens_sub, dv_sub], axis=1)
	main1 = np.concatenate([dens_main1_rep, v_main1], axis=1)
	dmain1 = np.concatenate([ddens_main1_rep, dv_main1], axis=1)
	main2 = np.concatenate([dens_main2_rep, v_main2], axis=1)
	dmain2 = np.concatenate([ddens_main2_rep, dv_main2], axis=1)
	cap = np.concatenate([dens_cap, v_cap], axis=1)
	dcap = np.concatenate([ddens_cap, dv_cap], axis=1)
	stack = [sub, main1, main2, cap]
	dstack = [dsub, dmain1, dmain2, dcap]
	
	return b, dens, ddens, v, dv, stack, dstack
	

def format_metadata(data_props, inst_props, sample_props, r):
	N, q_min, q_max, t_min = data_props
	wavelen, res, dres, log_bkg, dlog_bkg = inst_props
	sample_name, layer_names, rho, M, stack, dstack, prox = sample_props

	layers = ['(0) substrate', '(1) main 1', '(2) main 2', '(3) capping']
	properties = ['density (fu/A^3)', 'thickness (A)', 'roughness (A)', 'magnetization (uB/fu)']

	meta = {'PARAMETERS': {'sample': sample_name, 'layers': NoIndent(layer_names),
			'densities (g/cm^3)': NoIndent(rho), 'molar masses (g/mol)': NoIndent(M),
			'number of q-points': N, 'q_min (A^-1)': q_min, 'q_max (A^-1)': q_max,
			'wavelength (A)': wavelen, 't_min (A)': t_min, 'resolution': NoIndent([res, dres]),
			'background (log)': NoIndent([log_bkg, dlog_bkg]), 'r': r}}

	meta['STACK'] = {}
	for l in range(len(layers)):
		
		meta['STACK'][layers[l]] = {}
		for s in range(stack[l].shape[0]):
			if np.any(np.array([(prox[i][0] == l) & (prox[i][1] == s) for i in range(prox.shape[0])])):
				sublayer_key = '(' + str(s) + ') proximity'
			else: sublayer_key = '(' + str(s) + ')'
			
			meta['STACK'][layers[l]][sublayer_key] = {}
			for p in range(len(properties)):
				meta['STACK'][layers[l]][sublayer_key][properties[p]] = NoIndent([stack[l][s, p], dstack[l][s, p]])
	meta = json.dumps(meta, cls=CustomEncoder, sort_keys=True, indent=2)
	return meta


def merge_files(data_dir, tag, numfiles, header=None):
	filenames = []
	for k in range(numfiles):
		filenames +=  [data_dir + '/' + tag + '_' + str(k) + '.txt']

	with open(data_dir + '/' + tag + '.txt', 'w') as outfile:
		if header:
			outfile.write(','.join(header) + '\n')
		for fname in filenames:
			with open(fname) as infile:
				for line in infile:
					outfile.write(line)


def make_dirs(num_samples, sample_name, meta):
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

	num_tasks = num_samples
	delta = num_tasks//size
	start = rank*delta
	stop = start + delta
	if rank == size - 1:
		stop = num_tasks

	data_dir = 'results/' + sample_name
	if rank == 0:
		if not os.path.exists(data_dir):
			os.makedirs(data_dir)
	comm.barrier()

	dirs = next(os.walk(data_dir))[1]
	if len(dirs):
		idns = [int(d.split('_')[-1]) for d in dirs]
		idn = max(idns) + 1
	else:
		idn = 0
	comm.barrier()

	metafile = data_dir + '/set_' + str(idn) + '/metadata.txt'
	data_dir = data_dir + '/set_' + str(idn) + '/data'
	if rank == 0:
		os.makedirs(data_dir)
		print('data directory: ', data_dir)
		with open(metafile, 'w') as f:
			f.write(meta)
	comm.barrier()
	
	t0 = time.time()
	np.random.seed(rank)

	xdata_uu =  data_dir + '/xdata_uu_' + str(rank) + '.txt'
	xdata_dd =  data_dir + '/xdata_dd_' + str(rank) + '.txt'
	ydata =  data_dir + '/ydata_' + str(rank) + '.txt'

	return data_dir, [xdata_uu, xdata_dd, ydata], comm, rank, start, stop, size, t0


def clean_dirs(data_dir, data_files, comm, rank, size, t0, header):
	xdata_uu, xdata_dd, ydata = data_files

	comm.barrier()
	if rank == 0:
		print('time elapsed (min): ', (time.time() - t0)/60.)
		print('merging files')
		merge_files(data_dir, 'xdata_uu', size)
		merge_files(data_dir, 'xdata_dd', size)
		merge_files(data_dir, 'ydata', size, header)
		print('removing files')

	comm.barrier()
	os.remove(xdata_uu)
	os.remove(xdata_dd)
	os.remove(ydata)