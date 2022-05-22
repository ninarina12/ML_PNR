import numpy as np
import os
import sys
import copy

def blockPrint():
	sys.stdout = open(os.devnull, 'w')


def enablePrint(out=None):
	if out: sys.stdout = out
	else: sys.stdout = sys.__stdout__
               
out = sys.stdout
blockPrint() # suppress warnings on import
from genx.models import spec_nx as model
from genx.models.utils import UserVars, fp, fw, bc, bw
from genx.models.utils import create_fp, create_fw
enablePrint(out)


u = 1.66054 # conversion constant for density calculation

def build_inst(wavelen, Ibkg=0.0, res=0.0005):
	inst = model.Instrument(Ibkg=Ibkg, res=res, footype='no corr', I0=1.0, samplelen=10.0, 
							probe='neutron pol', beamw=0.01, restype='full conv and varying res.',
							tthoff=0.0, pol='uu', coords='q', resintrange=2, wavelength=wavelen,
							respoints=5, incangle=0.0,)
	inst_fp = create_fp(inst.wavelength)
	inst_fw = create_fw(inst.wavelength)
	fp.set_wavelength(inst.wavelength)
	fw.set_wavelength(inst.wavelength)
	cp = UserVars()

	return inst


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


def simulate(q, inst, b, dens, v, rres, bkg):
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
	# saved units: density (fu/nm^3), thickness (nm), roughness (nm), magnetization (uB/fu)
	stack = np.concatenate([sub, main1, main2, cap], axis=0)
	dens, d, s, magn = list(1E3*stack[:,0]), list(0.1*stack[:,1]), list(0.1*stack[:,2]), list(stack[:,3])
	p = dens + d + s + magn + [rres, bkg]

	# build sample and compute spectra
	sample = build_stack(b, sub, main1, main2, cap)
	I = reflectivity_sim(q, inst, sample, rres, bkg)
	return p, I


def reflectivity_sim(q, inst, sample, res=0.00055, Ibkg=1e-6):
	I = []
	inst.setRes(res)
	inst.setIbkg(Ibkg)
	inst.setPol('uu')
	I.append(sample.SimSpecular(q, inst))
	inst.setPol('dd')
	I.append(sample.SimSpecular(q, inst))
	return I


def SLD_sim(q, inst, sample, res=0.00055, Ibkg=1e-6):
	SLD = []
	inst.setRes(res)
	inst.setIbkg(Ibkg)
	inst.setPol('uu')
	SLD.append(sample.SimSLD(None, None, inst))
	return SLD[0]


def run(data_files, start, stop, q, t_min, wavelen, res, dres, log_bkg, dlog_bkg, b, dens_avg, ddens, v_avg, dv, prox, r):
	xdata_uu, xdata_dd, ydata = data_files
	inst = build_inst(wavelen)

	with open(xdata_uu, 'w') as fx1, open(xdata_dd, 'w') as fx2, open(ydata, 'w') as fy:
		for k in range(start, stop):
		
			# randomly sample instrument properties
			rres = res + 2*dres*np.random.random_sample() - dres
			rlog_bkg = log_bkg + 2*dlog_bkg*np.random.random_sample() - dlog_bkg
			bkg = 10*np.random.random_sample()*10**rlog_bkg

			# randomly sample sample densities
			dens = copy.deepcopy(dens_avg)
			for i in range(len(dens)):
				dens[i] = dens_avg[i] + 2*ddens[i]*np.random.random_sample(size=dens[i].shape) - ddens[i]

			# randomly sample sample variables
			v = copy.deepcopy(v_avg)
			for i in range(len(v)):
				v[i] = v_avg[i] + 2*dv[i]*np.random.random_sample(size=v[i].shape) - dv[i]

			# interfacial magnetism present or not (above or below thickness threshold)
			m = [int(v[prox[i][0]][prox[i][1],0] >= t_min) for i in range(prox.shape[0])]

			# proximity magnetism cannot exceed FM magnetization, if present
			if r and m[0]:
				vmin = v_avg[prox[0][0]][prox[0][1],2] - dv[prox[0][0]][prox[0][1],2]

				# note: vmax must be weighted by the density ratio to compare magnetizations in units of emu/cm^3 (not per f.u.)
				i_mag = prox[0][0] - int(prox[0][1] == 0) + int(prox[0][1] > 0)
				r_dens = dens[i_mag][0,0]/dens[prox[0][0]][0,0]

				vmax = min(v_avg[prox[0][0]][prox[0][1],2] + dv[prox[0][0]][prox[0][1],2], r*v[i_mag][0,2]*r_dens)

				# ensure that vmax > vmin
				while vmax < vmin:
					v[i_mag][0,2] = v_avg[i_mag][0,2] + 2*dv[i_mag][0,2]*np.random.random_sample() - dv[i_mag][0,2]
					vmax = min(v_avg[prox[0][0]][prox[0][1],2] + dv[prox[0][0]][prox[0][1],2], r*v[i_mag][0,2]*r_dens)
				
				v[prox[0][0]][prox[0][1],2] = (vmax - vmin)*np.random.random_sample() + vmin

			# if interfacial magnetism absent, re-allocate film sublayer thickness to bulk sublayer
			for i in range(prox.shape[0]):
				i_bulk = prox[i][1] + int(prox[i][1] == 0) - int(prox[i][1] > 0)
				v[prox[i][0]][i_bulk,0] += (1 - m[i])*v[prox[i][0]][prox[i][1],0]
				v[prox[i][0]][prox[i][1],0] *= m[i]
				v[prox[i][0]][prox[i][1],2] *= m[i]

			# simulate
			p, I = simulate(q, inst, b, dens, v, rres, bkg)

			# write to files
			fx1.write(' '.join([str(j) for j in I[0]]) + '\n')
			fx2.write(' '.join([str(j) for j in I[1]]) + '\n')
			fy.write(','.join([str(j) for j in p]) + '\n')