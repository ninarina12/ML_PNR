import numpy as np
import pandas as pd
import os, sys

from scipy.interpolate import interp1d
from plot_imports import *

def blockPrint():
	sys.stdout = open(os.devnull, 'w')

def enablePrint():
	sys.stdout = sys.__stdout__

blockPrint() # avoid printing non-essential warnings on import
from genx.models import spec_nx as model
from genx.models.utils import UserVars, fp, fw, bc, bw
from genx.models.utils import create_fp, create_fw
enablePrint()

# b = neutron scattering length
# dens = density (g/cm^3)/u/molar mass of compound (g/mol)
# d = layer thickness (A)
# s = roughness of layer top surface (A)
# magn = magnetization (uB)

def Al2O3_Bi2Se3_EuS_aAl2O3(y_pred, y_header):
	layer_names = [r'$\alpha-Al_2O_3$', '$Bi_2Se_3$', 'EuS', '$a-Al_2O_3$']

	# neutron scattering lengths
	b_sub = bc.Al*2+bc.O*3
	b_main1 = bc.Bi*2+bc.Se*3
	b_main2 = bc.Eu+bc.S
	b_cap = [bc.Al*2+bc.O*3]

	sub = np.array([y_pred[y_header.index('dens_sub')]*1e-3] + [0., 1., 0.])
	
	main1 = np.array([[y_pred[y_header.index('dens_TI')]*1e-3, y_pred[y_header.index('dens_TI')]*1e-3],
	                  [y_pred[y_header.index('d_TI')]*10., y_pred[y_header.index('d_prox')]*10.],
	                  [1., 1.],
	                  [0., y_pred[y_header.index('magn_prox')]]]).T

	main2 = np.array([[y_pred[y_header.index('dens_FM')]*1e-3],
	                  [y_pred[y_header.index('d_FM')]*10.],
	                  [1.],
	                  [y_pred[y_header.index('magn_FM')]]]).T

	cap = np.array([[y_pred[y_header.index('dens_cap')]*1e-3],
	                [y_pred[y_header.index('d_cap')]*10.],
	                [1.],
	                [0.]]).T

	b = [b_sub, b_main1, b_main2, b_cap]
	
	layer_bounds = [0, sum(main1[:,1]), sum(main1[:,1]) + sum(main2[:,1])]
	for i in range(len(cap)):
		layer_bounds += [layer_bounds[-1] + cap[i,1]]

	return layer_names, layer_bounds, b, sub, main1, main2, cap


def Al2O3_Cr2O3_BiSb2Te3_Te_TeO2(y_pred, y_header):
	layer_names = [r'$\alpha-Al_2O_3$', '$Cr_2O_3$', '$(Bi,Sb)_2Te_3$', 'Te/$TeO_2$']

	# neutron scattering lengths
	b_sub = bc.Al*2+bc.O*3
	b_main1 = bc.Cr*2+bc.O*3
	b_main2 = bc.Bi*0.4+bc.Sb*1.6+bc.Te*3
	b_cap = [bc.Te, bc.Te+bc.O*2]

	sub = np.array([y_pred[y_header.index('dens_sub')]*1e-3] + [0., 1., 0.])
	
	main1 = np.array([[y_pred[y_header.index('dens_AFM')]*1e-3, y_pred[y_header.index('dens_AFM')]*1e-3],
	                  [y_pred[y_header.index('d_AFM')]*10., y_pred[y_header.index('d_iAFM')]*10.],
	                  [1., 1.],
	                  [0., y_pred[y_header.index('magn_iAFM')]]]).T
	
	main2 = np.array([[y_pred[y_header.index('dens_TI')]*1e-3, y_pred[y_header.index('dens_TI')]*1e-3],
	                  [y_pred[y_header.index('d_prox')]*10., y_pred[y_header.index('d_TI')]*10.],
	                  [1., 1.],
	                  [y_pred[y_header.index('magn_prox')], 0.]]).T

	cap = np.array([[y_pred[y_header.index('dens_cap')]*1e-3, y_pred[y_header.index('dens_ox')]*1e-3],
	                [y_pred[y_header.index('d_cap')]*10., y_pred[y_header.index('d_ox')]*10.],
	                [1., 1.],
	                [0., 0.]]).T

	b = [b_sub, b_main1, b_main2, b_cap]
	layer_bounds = [0, sum(main1[:,1]), sum(main1[:,1]) + sum(main2[:,1]),
					sum(main1[:,1]) + sum(main2[:,1]) + sum(cap[:,1])]

	return layer_names, layer_bounds, b, sub, main1, main2, cap


# simulation
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

	dens, d, s, magn = sub
	Sub = model.Layer(b=b_sub, d=d, dens=dens, sigma=s, magn_ang=0.0, xs_ai=0.0, magn=magn)

	Cap = model.Stack(Layers=Cap, Repetitions=1)
	Main2 = model.Stack(Layers=Main2, Repetitions=1)
	Main1 = model.Stack(Layers=Main1, Repetitions=1)

	sample = model.Sample(Stacks=[Main1, Main2, Cap], Ambient=Amb, Substrate=Sub)
	return sample


def reflectivity_sim(q, inst, sample):
	I = []
	inst.setPol('uu')
	I.append(sample.SimSpecular(q, inst))
	inst.setPol('dd')
	I.append(sample.SimSpecular(q, inst))
	return I


def SLD_sim(q, inst, sample):
	SLD = []
	inst.setPol('uu')
	SLD.append(sample.SimSLD(None, None, inst))
	return SLD[0]


def plot_SLD(image_dir, sample_name, y_pred, y_header, title, scale=0, y_lims=None):
	# constants
	if sample_name == 'BiSe10_EuS5':
		structure, q_max, wavelen, res, Ibkg = 'Al2O3_Bi2Se3_EuS_aAl2O3', 0.13, 4.75, 5e-4, 3.35e-6

	else:
		structure, q_max, wavelen, res, Ibkg = 'Al2O3_Cr2O3_BiSb2Te3_Te_TeO2', 0.175, 5.35, 1e-3, 1e-6

	N = 256                                                 # number of q points for reflectivity plot
	q_min = 0.01
	q = np.linspace(q_min, q_max, N)                        # q points at which to sample reflectivity (A^-1)

	# instrument properties

	inst = model.Instrument(Ibkg=Ibkg, res=res, footype='no corr', I0=1.0, samplelen=10.0, 
							probe='neutron pol', beamw=0.01, restype='full conv and varying res.',
							tthoff=0.0, pol='uu', coords='q', resintrange=2, wavelength=wavelen,
							respoints=5, incangle=0.0,)
	inst_fp = create_fp(inst.wavelength); inst_fw = create_fw(inst.wavelength)
	fp.set_wavelength(inst.wavelength); fw.set_wavelength(inst.wavelength)
	cp = UserVars()

	layer_names, layer_bounds, b, sub, main1, main2, cap = eval(structure)(y_pred, y_header)

	# build sample and compute spectra
	sample = build_stack(b, sub, main1, main2, cap)

	sld = SLD_sim(q, inst, sample)

	# resample to N points
	z = np.linspace(sld['z'].min(), sld['z'].max(), N)
	re = interp1d(sld['z'], sld['Re non-mag'])(z)
	im = interp1d(sld['z'], sld['Im non-mag'])(z)
	mag = interp1d(sld['z'], sld['mag'])(z)
	z *= 0.1
	layer_bounds = 0.1*np.array(layer_bounds)

	# set up figure and axes
	fig, ax = plt.subplots(figsize=(5,4.5))
	prop.set_size(18)
	lprop = prop.copy()
	lprop.set_size(14)

	# plot SLD
	if scale:
		mag *= scale
		im *= scale
		tag = ' x ' + str(scale)
	else:
		tag = ''

	alpha = 0.5
	ax.plot(z, re, color=palette[1], zorder=1)
	ax.plot(z, mag, color=palette[3], zorder=2)
	ax.plot(z, im, color=palette[-1], zorder=3)
	ax.fill_between(z, re, color=palette[1], alpha=alpha - 0.2, label='NSLD', zorder=1)
	ax.fill_between(z, mag, color=palette[3], alpha=alpha, label='MSLD' + tag, zorder=2)
	ax.fill_between(z, im, color=palette[-1], alpha=alpha, label='ASLD' + tag, zorder=3)
	for i in range(len(layer_bounds)):
		ax.axvline(layer_bounds[i], linestyle='--', color='#ADABA4')

	ax.legend(prop=lprop, edgecolor='white', framealpha=1, loc='upper right')
	name_pos = (layer_bounds[:-1] + layer_bounds[1:])/2.
	for i, name in enumerate(layer_names[1:]):
		ax.text(name_pos[i], 1., name, color='#262626', fontproperties=lprop, ha='center', va='center')

	if not y_lims: y_lims = [im.min() - 0.5, re.max() + 0.5]
	format_axis(ax, 'z (nm)', 'SLD (10$^{-4}$/nm$^2$)', prop, xlims=[z.min(), z.max()], ylims=y_lims)
	fig.tight_layout()
	fig.subplots_adjust(wspace=0.25)
	fig.savefig(image_dir + '/sld_' + title + '.pdf', dpi='figure')
	return y_lims
