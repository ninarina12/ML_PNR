import numpy as np
import pandas as pd
import time
import os, sys
import copy
from scipy.interpolate import interp1d
from mpi4py import MPI

def blockPrint():
	sys.stdout = open(os.devnull, 'w')

def enablePrint():
	sys.stdout = sys.__stdout__

blockPrint() # suppress warnings on import
from genx.models import spec_nx as model
from genx.models.utils import UserVars, fp, fw, bc, bw
from genx.models.utils import create_fp, create_fw
enablePrint()

############################################### formatting helpers ##############################################

from _ctypes import PyObj_FromPtr
import json
import re

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
	
def format_metadata(sample_name, structure, layer_names, N, q_min, q_max, wavelen,
					res, dres, log_bkg, dlog_bkg, stack, dstack, prox):

	layers = ['(0) substrate', '(1) main 1', '(2) main 2', '(3) capping']
	properties = ['density (fu/A^3)', 'thickness (A)', 'roughness (A)', 'magnetization (uB)']

	meta = {'PARAMETERS': {'sample': sample_name, 'structure': structure, 'layers': NoIndent(layer_names),
			'number of q-points': N, 'q_min (A^-1)': q_min, 'q_max (A^-1)': q_max, 'wavelength (A)': wavelen,
			'resolution': NoIndent([res, dres]), 'background (log)': NoIndent([log_bkg, dlog_bkg])}}

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

def read_exp(file_name):
	df = pd.read_csv(file_name, usecols=[0,1,2,3], names=['Q','R','dR','dQ'], skiprows=24, sep='\t', dtype=float)
	return df

################################################### constants ###################################################

# name of experiment sample, name of function to build sample stack, maximum q (A^-1), neutron wavelength (A)
#sample_name, structure, q_max, wavelen = 'BiSe10_EuS5', 'Al2O3_Bi2Se3_EuS_aAl2O3', 0.13, 4.75
sample_name, structure, q_max, wavelen = 'CrO20_BiSbTe20', 'Al2O3_Cr2O3_BiSb2Te3_Te_TeO2', 0.17, 5.35

N = 256                                                 # number of q points for reflectivity plot
q_min = 0.01
q = np.linspace(q_min, q_max, N)                        # q points at which to sample reflectivity (A^-1)

res = 0.5*(0.001 + 0.0001)								# instrument resolution
dres = 0.5*(0.001 - 0.0001)								# uncertainty in instrument resolution

log_bkg = -7											# instrument background (log10)
dlog_bkg = 3											# uncertainty in instrument background (log10)

num_samples = 100000                                    # number of examples to generate
u = 1.66054                                             # conversion constant for density calculation

############################################# instrument properties #############################################

inst = model.Instrument(Ibkg=0.0, res=0.001, footype='no corr', I0=1.0, samplelen=10.0, 
						probe='neutron pol', beamw=0.01, restype='full conv and varying res.',
						tthoff=0.0, pol='uu', coords='q', resintrange=2, wavelength=wavelen,
						respoints=5, incangle=0.0,)
inst_fp = create_fp(inst.wavelength); inst_fw = create_fw(inst.wavelength)
fp.set_wavelength(inst.wavelength); fw.set_wavelength(inst.wavelength)
cp = UserVars()

############################################# experiment properties #############################################

exp_names = next(os.walk('experiments/' + sample_name))[1]
exp_names = [exp_names[j] for j in np.argsort([int(i[:i.index('K')]) for i in exp_names])]

# we will use the lowest temperature measurement for noise estimates
df = read_exp('experiments/' + sample_name + '/' + exp_names[0] + '/x_uu.dat')
q_exp = df['Q'].values

#################################################### samples ####################################################

# variables (v) = [d, s, magn]

# b = neutron scattering length
# dens = density (g/cm^3)/u/molar mass of compound (g/mol)
# d = layer thickness (A)
# s = roughness of layer top surface (A)
# magn = magnetization (uB)

def Al2O3_Bi2Se3_EuS_aAl2O3():
	# layers
	layer_names = ['$Al_2O_3$', '$Bi_2Se_3$', '$Bi_2Se_3$\ninterface', 'EuS', r'$\alpha-Al_2O_3$']

	# sub
	b_sub = bc.Al*2+bc.O*3
	dens_sub = np.array([[3.95/u/101.96]])
	ddens_sub = 0.05*dens_sub
	v_sub = np.array([[0, 4., 0]])
	dv_sub = np.array([[0, 4., 0]])

	# main1 (2 layers)
	b_main1 = bc.Bi*2+bc.Se*3
	dens_main1 = np.array([[6.82/u/654.8]])
	ddens_main1 = 0.2*dens_main1
	v_main1 = np.array([[100., 7., 0], [20., 5., 3.]])
	dv_main1 = np.array([[20., 7., 0], [10., 5., 2.]])
	
	# main2
	b_main2 = bc.Eu+bc.S
	dens_main2 = np.array([[5.75/u/184.03]])
	ddens_main2 = 0.2*dens_main2
	v_main2 = np.array([[60., 8., 4.]])
	dv_main2 = np.array([[20., 4., 4.]])

	# cap
	b_cap = [bc.Al*2+bc.O*3]
	dens_cap = np.array([[2.75/u/101.96]])
	ddens_cap = np.array([[0.5]])*dens_cap
	v_cap = np.array([[100., 10., 0]])
	dv_cap = np.array([[30., 4., 0]])

	# proximity layer(s)
	prox = np.array([[1,1]])		# main1, sublayer 2

	header = ['dens_sub', 'dens_TI', 'dens_TI', 'dens_FM', 'dens_cap',
			  'd_sub', 'd_TI', 'd_prox', 'd_FM', 'd_cap',
			  's_sub', 's_TI', 's_prox', 's_FM', 's_cap',
			  'magn_sub', 'magn_TI', 'magn_prox', 'magn_FM', 'magn_cap']

	# format parameters
	b, dens, ddens, v, dv, stack, dstack = format_parameters(b_sub, b_main1, b_main2, b_cap,
															 dens_sub, dens_main1, dens_main2, dens_cap,
															 ddens_sub, ddens_main1, ddens_main2,
															 ddens_cap, v_sub, v_main1, v_main2, v_cap,
															 dv_sub, dv_main1, dv_main2, dv_cap)
	
	# create metadata
	meta = format_metadata(sample_name, structure, layer_names, N, q_min, q_max, wavelen,
						   res, dres, log_bkg, dlog_bkg, stack, dstack, prox)

	return b, dens, ddens, v, dv, prox, header, meta

def Al2O3_Cr2O3_BiSb2Te3_Te_TeO2():
	# layers
	layer_names = ['$Al_2O_3$', '$Cr_2O_3$', '$Cr_2O_3$\ninterface', 'TI\ninterface', 'TI', 'Te', '$TeO_2$']

	# sub
	b_sub = bc.Al*2+bc.O*3
	dens_sub = np.array([[3.95/u/101.96]])
	ddens_sub = 0.05*dens_sub
	v_sub = np.array([[0, 4., 0]])
	dv_sub = np.array([[0, 4., 0]])

	# main1 (2 layers)
	b_main1 = bc.Cr*2+bc.O*3
	dens_main1 = np.array([[5.22/u/151.99]])
	ddens_main1 = 0.3*dens_main1
	v_main1 = np.array([[200., 10., 0], [20., 5., 0.6]])
	dv_main1 = np.array([[20., 8., 0], [10., 5., 0.4]])
	
	# main2 (2 layers)
	b_main2 = bc.Bi*0.4+bc.Sb*1.6+bc.Te*3
	dens_main2 = np.array([[0.2*(7.7/u/800.76) + 0.8*(6.5/u/626.32)]])
	ddens_main2 = 0.3*dens_main2
	v_main2 = np.array([[20., 5., 0.6], [180., 20., 0.]])
	dv_main2 = np.array([[10., 5., 0.4], [20., 10., 0.]])

	# cap
	b_cap = [bc.Te, bc.Te+bc.O*2]
	dens_cap = np.array([[6.24/u/127.6], [5.67/u/159.6]])
	ddens_cap = np.array([[0.5], [0.5]])*dens_cap
	v_cap = np.array([[90., 20., 0], [25., 25., 0]])
	dv_cap = np.array([[20., 10., 0], [10., 10., 0]])

	# proximity layer(s)
	prox = np.array([[1,1],			# main1, sublayer 2
					 [2,0]])		# main2, sublayer 1

	header = ['dens_sub', 'dens_AFM', 'dens_AFM', 'dens_TI', 'dens_TI', 'dens_cap', 'dens_ox',
			  'd_sub', 'd_AFM', 'd_iAFM', 'd_prox', 'd_TI', 'd_cap', 'd_ox',
			  's_sub', 's_AFM', 's_iAFM', 's_prox', 's_TI', 's_cap', 's_ox',
			  'magn_sub', 'magn_AFM', 'magn_iAFM', 'magn_prox', 'magn_TI', 'magn_cap', 'magn_ox']

	# format parameters
	b, dens, ddens, v, dv, stack, dstack = format_parameters(b_sub, b_main1, b_main2, b_cap,
															 dens_sub, dens_main1, dens_main2, dens_cap,
															 ddens_sub, ddens_main1, ddens_main2,
															 ddens_cap, v_sub, v_main1, v_main2, v_cap,
															 dv_sub, dv_main1, dv_main2, dv_cap)

	# create metadata
	meta = format_metadata(sample_name, structure, layer_names, N, q_min, q_max, wavelen,
						   res, dres, log_bkg, dlog_bkg, stack, dstack, prox)

	return b, dens, ddens, v, dv, prox, header, meta

################################################## simulation ###################################################

def build_stack(b, sub, main1, main2, cap):
	b_sub, b_main1, b_main2, b_cap = b

	Amb = model.Layer(b=0j, d=0.0, f=(1e-20+1e-20j), dens=1.0, sigma=0.0, magn_ang=0.0, xs_ai=0.0, magn=0.0)
	
	Cap = []
	for i in range(cap.shape[0]):
		dens, d, s, magn = cap[i]
		Cap += [model.Layer(b=b_cap[i], d=d, dens=dens, sigma=s, magn_ang=0.0, xs_ai=0.0, magn=magn)]

	Main2 = []
	for i in range(main2.shape[0]):
		dens, d, s, magn = main2[i]
		Main2 += [model.Layer(b=b_main2, d=d, dens=dens, sigma=s, magn_ang=0.0, xs_ai=0.0, magn=magn)]
	
	Main1 = []
	for i in range(main1.shape[0]):
		dens, d, s, magn = main1[i]
		Main1 += [model.Layer(b=b_main1, d=d, dens=dens, sigma=s, magn_ang=0.0, xs_ai=0.0, magn=magn)]

	dens, d, s, magn = sub[0]
	Sub = model.Layer(b=b_sub, d=d, dens=dens, sigma=s, magn_ang=0.0, xs_ai=0.0, magn=magn)

	Cap = model.Stack(Layers=Cap, Repetitions=1)
	Main2 = model.Stack(Layers=Main2, Repetitions=1)
	Main1 = model.Stack(Layers=Main1, Repetitions=1)

	sample = model.Sample(Stacks=[Main1, Main2, Cap], Ambient=Amb, Substrate=Sub)
	return sample

def reflectivity_sim(q, inst, sample, res, Ibkg):
	I = []
	inst.setRes(res)
	inst.setIbkg(Ibkg)
	inst.setPol('uu')
	I.append(sample.SimSpecular(q, inst))
	inst.setPol('dd')
	I.append(sample.SimSpecular(q, inst))
	return I

def SLD_sim(q, inst, sample, res, Ibkg):
	SLD = []
	inst.setRes(res)
	inst.setIbkg(Ibkg)
	inst.setPol('uu')
	SLD.append(sample.SimSLD(None, None, inst))
	return SLD[0]

def spin_asymmetry(I):
	return (I[0]-I[1])/(I[0]+I[1])

def merge_files(datafile, tag, numfiles, header=None):
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

################################################ data properties ################################################

b, dens_avg, ddens, v_avg, dv, prox, header, meta = eval(structure)()

################################################ run in parallel ################################################

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
	t0 = time.time()
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

np.random.seed(rank)
xdata_uu =  data_dir + '/xdata_uu_' + str(rank) + '.txt'
xdata_dd =  data_dir + '/xdata_dd_' + str(rank) + '.txt'
ydata =  data_dir + '/ydata_' + str(rank) + '.txt'
zdata =  data_dir + '/zdata_' + str(rank) + '.txt'
nucdata =  data_dir + '/nucdata_' + str(rank) + '.txt'
absdata =  data_dir + '/absdata_' + str(rank) + '.txt'
magdata =  data_dir + '/magdata_' + str(rank) + '.txt'
spindata =  data_dir + '/spindata_' + str(rank) + '.txt'

# for direct comparison to experiment (e.g. for adding noise), also save examples sampled at q_exp
xdata_uu_exp =  data_dir + '/xdata_uu_exp_' + str(rank) + '.txt'
xdata_dd_exp =  data_dir + '/xdata_dd_exp_' + str(rank) + '.txt'

with open(xdata_uu, 'w') as fx1, open(xdata_dd, 'w') as fx2, open(ydata, 'w') as fy, open(zdata, 'w') as fz, \
	 open(nucdata, 'w') as fn, open(absdata, 'w') as fa, open(magdata, 'w') as fm, open(spindata, 'w') as fs, \
	 open(xdata_uu_exp, 'w') as fx1_exp, open(xdata_dd_exp, 'w') as fx2_exp:
	
	for k in range(start, stop):
		
		# randomly sample instrument properties
		rres = res + 2*dres*np.random.random_sample() - dres
		rlog_bkg = log_bkg + 2*dlog_bkg*np.random.random_sample() - dlog_bkg
		
		# randomly sample sample densities
		dens = copy.deepcopy(dens_avg)
		for i in range(len(dens)):
			dens[i] = dens_avg[i] + 2*ddens[i]*np.random.random_sample(size=dens[i].shape) - ddens[i]

		# randomly sample sample variables
		v = copy.deepcopy(v_avg)
		for i in range(len(v)):
			v[i] = v_avg[i] + 2*dv[i]*np.random.random_sample(size=v[i].shape) - dv[i]

		# if FM sample, proximity magnetism cannot exceed FM magnetization
		if prox.shape[0] == 1:
			vmin = v_avg[prox[0][0]][prox[0][1],2] - dv[prox[0][0]][prox[0][1],2]
			if prox[0][1]:
				# magnetic layer is at prox[0][0]+1
				vmax = min(v_avg[prox[0][0]][prox[0][1],2] + dv[prox[0][0]][prox[0][1],2], v[prox[0][0]+1][0,2])
			else:
				# magnetic layer is at prox[0][0]-1
				vmax = min(v_avg[prox[0][0]][prox[0][1],2] + dv[prox[0][0]][prox[0][1],2], v[prox[0][0]-1][0,2])

			# proximity present or absent
			m = np.random.randint(2, size=(1,1))

			if m:
				# if m > 0, ensure that vmax > vmin
				while vmax < vmin:
					if prox[0][1]:
						# magnetic layer is at prox[0][0]+1
						v[prox[0][0]+1][0,2] = v_avg[prox[0][0]+1][0,2] + \
											   2*dv[prox[0][0]+1][0,2]*np.random.random_sample() \
											   - dv[prox[0][0]+1][0,2]
						vmax = min(v_avg[prox[0][0]][prox[0][1],2] + dv[prox[0][0]][prox[0][1],2], v[prox[0][0]+1][0,2])
					else:
						# magnetic layer is at prox[0][0]-1
						v[prox[0][0]-1][0,2] = v_avg[prox[0][0]-1][0,2] + \
											   2*dv[prox[0][0]-1][0,2]*np.random.random_sample() \
											   - dv[prox[0][0]-1][0,2]
						vmax = min(v_avg[prox[0][0]][prox[0][1],2] + dv[prox[0][0]][prox[0][1],2], v[prox[0][0]-1][0,2])

			v[prox[0][0]][prox[0][1],2] = m[0]*((vmax - vmin)*np.random.random_sample() + vmin)

		else:
			# proximity present or absent
			m = np.random.randint(2, size=(prox.shape[0],1))

			for i in range(prox.shape[0]):
				v[prox[i][0]][prox[i][1],2] *= m[i]

		# if proximity absent, re-allocate film sublayer thickness to bulk sublayer
		for i in range(prox.shape[0]):
			i_bulk = prox[i][1] + int(prox[i][1] == 0) - int(prox[i][1] > 0)
			v[prox[i][0]][i_bulk,0] += (1 - m[i])*v[prox[i][0]][prox[i][1],0]
			v[prox[i][0]][prox[i][1],0] *= m[i]

		# unpack values
		dens_sub, dens_main1, dens_main2, dens_cap = dens
		v_sub, v_main1, v_main2, v_cap = v

		# replicate properties for subdivided layers
		dens_main1 = np.repeat(dens_main1, v_main1.shape[0], axis=0)
		dens_main2 = np.repeat(dens_main2, v_main2.shape[0], axis=0)

		# assemble all parameters
		sub = np.concatenate([dens_sub, v_sub], axis=1)
		main1 = np.concatenate([dens_main1, v_main1], axis=1)
		main2 = np.concatenate([dens_main2, v_main2], axis=1)
		cap = np.concatenate([dens_cap, v_cap], axis=1)

		# organize and convert parameters for saving
		stack = np.concatenate([sub, main1, main2, cap], axis=0)
		dens, d, s, magn = list(1E3*stack[:,0]), list(0.1*stack[:,1]), list(0.1*stack[:,2]), list(stack[:,3])
		p = dens + d + s + magn
		
		# randomly sample background
		bkg = 10*np.random.random_sample()*10**rlog_bkg

		# build sample and compute spectra
		sample = build_stack(b, sub, main1, main2, cap)
		I = reflectivity_sim(q, inst, sample, rres, bkg)
		I_exp = reflectivity_sim(q_exp, inst, sample, rres, bkg)
		sld = SLD_sim(q, inst, sample, rres, bkg)
		sa = spin_asymmetry(I)
		
		# resample to N points
		z = np.linspace(sld['z'].min(), sld['z'].max(), N)
		re = interp1d(sld['z'], sld['Re non-mag'])(z)
		im = interp1d(sld['z'], sld['Im non-mag'])(z)
		mag = interp1d(sld['z'], sld['mag'])(z)

		# write to files
		fx1.write(' '.join([str(j) for j in I[0]]) + '\n')
		fx2.write(' '.join([str(j) for j in I[1]]) + '\n')
		fy.write(','.join([str(j) for j in p]) + '\n')
		fz.write(' '.join([str(j) for j in 0.1*z]) + '\n')
		fn.write(' '.join([str(j) for j in re]) + '\n')
		fa.write(' '.join([str(j) for j in im]) + '\n')
		fm.write(' '.join([str(j) for j in mag]) + '\n')
		fs.write(' '.join([str(j) for j in sa]) + '\n')
		fx1_exp.write(' '.join([str(j) for j in I_exp[0]]) + '\n')
		fx2_exp.write(' '.join([str(j) for j in I_exp[1]]) + '\n')

comm.barrier()
if rank == 0:
	print('time elapsed (min): ', (time.time() - t0)/60.)
	print('merging files')
	merge_files(data_dir, 'xdata_uu', size)
	merge_files(data_dir, 'xdata_dd', size)
	merge_files(data_dir, 'ydata', size, header)
	merge_files(data_dir, 'zdata', size)
	merge_files(data_dir, 'nucdata', size)
	merge_files(data_dir, 'absdata', size)
	merge_files(data_dir, 'magdata', size)
	merge_files(data_dir, 'spindata', size)
	merge_files(data_dir, 'xdata_uu_exp', size)
	merge_files(data_dir, 'xdata_dd_exp', size)
	print('removing files')

comm.barrier()
os.remove(xdata_uu)
os.remove(xdata_dd)
os.remove(ydata)
os.remove(zdata)
os.remove(nucdata)
os.remove(absdata)
os.remove(magdata)
os.remove(spindata)
os.remove(xdata_uu_exp)
os.remove(xdata_dd_exp)
