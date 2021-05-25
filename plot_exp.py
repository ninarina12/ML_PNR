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

######################################## constants ########################################

# name of experiment sample, name of function to build sample stack, maximum q (A^-1), neutron wavelength (A)
sample_name, structure, q_max, wavelen = 'BiSe10_EuS5', 'Al2O3_Bi2Se3_EuS_aAl2O3', 0.13, 4.75
#sample_name, structure, q_max, wavelen = 'CrO20_BiSbTe20', 'Al2O3_Cr2O3_BiSb2Te3_Te_TeO2', 0.18, 5.35

T = 4													# temperature of experiment sample

N = 256                                                 # number of q points for reflectivity plot
q_min = 0.01
q = np.linspace(q_min, q_max, N)                        # q points at which to sample reflectivity (A^-1)

res = 5e-4												# instrument resolution
Ibkg = 3.35e-6											# instrument background

################################## instrument properties ##################################

inst = model.Instrument(Ibkg=Ibkg, res=res, footype='no corr', I0=1.0, samplelen=10.0, 
						probe='neutron pol', beamw=0.01, restype='full conv and varying res.',
						tthoff=0.0, pol='uu', coords='q', resintrange=2, wavelength=wavelen,
						respoints=5, incangle=0.0,)
inst_fp = create_fp(inst.wavelength); inst_fw = create_fw(inst.wavelength)
fp.set_wavelength(inst.wavelength); fw.set_wavelength(inst.wavelength)
cp = UserVars()

################################## experiment properties ##################################

def read_exp(file_name):
	df = pd.read_csv(file_name, usecols=[0,1,2,3], names=['Q','R','dR','dQ'], skiprows=24, sep='\t', dtype=float)
	return df

df_uu = read_exp('experiments/' + sample_name + '/' + str(T) + 'K/x_uu.dat')
df_dd = read_exp('experiments/' + sample_name + '/' + str(T) + 'K/x_dd.dat')

######################################### samples #########################################

# b = neutron scattering length
# dens = density (g/cm^3)/u/molar mass of compound (g/mol)
# d = layer thickness (A)
# s = roughness of layer top surface (A)
# magn = magnetization (uB)

def Al2O3_Bi2Se3_EuS_aAl2O3():
	layer_names = ['$Al_2O_3$', '$Bi_2Se_3$', 'EuS', r'$\alpha-Al_2O_3$']

	# sub
	b_sub = bc.Al*2+bc.O*3
	dens_sub = 0.0228708537774
	d_sub = 0
	s_sub = 6.0835001525
	magn_sub = 0
	sub = np.array([dens_sub, d_sub, s_sub, magn_sub])

	# main1 (2 layers)
	b_main1 = bc.Bi*2+bc.Se*3
	dens_main1 = [0.00642517262189, 0.00692682110541]
	d_main1 = [102.027429435, 0.178024121913]
	s_main1 = [1.64882334832, 7.64504806905]
	magn_main1 = [0, 4.27340641208]
	main1 = np.array([dens_main1, d_main1, s_main1, magn_main1]).T

	# main2
	b_main2 = bc.Eu+bc.S
	dens_main2 = [0.0186309206731]
	d_main2 = [59.7238601945]
	s_main2 = [7.77929046573]
	magn_main2 = [6.45998595423]
	main2 = np.array([dens_main2, d_main2, s_main2, magn_main2]).T

	# cap
	b_cap = [bc.Al*2+bc.O*3]
	dens_cap = [0.0160136391694]
	d_cap = [95.3159134224]
	s_cap = [11.7824468145]
	magn_cap = [0]
	cap = np.array([dens_cap, d_cap, s_cap, magn_cap]).T

	b = [b_sub, b_main1, b_main2, b_cap]
	
	layer_bounds = [0, sum(d_main1), sum(d_main1) + sum(d_main2)]
	for i in range(len(d_cap)):
		layer_bounds += [layer_bounds[-1] + d_cap[i]]

	return layer_names, layer_bounds, b, sub, main1, main2, cap

def Al2O3_Cr2O3_BiSb2Te3_Te_TeO2():
	# layers
	layer_names = ['$Al_2O_3$', '$Cr_2O_3$', '$Cr_2O_3$\ninterface', 'TI\ninterface', 'TI', 'Te', '$TeO_2$']

####################################### simulation ########################################

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

def spin_asymmetry(I):
	return (I[0]-I[1])/(I[0]+I[1])

######################################### script ##########################################

layer_names, layer_bounds, b, sub, main1, main2, cap = eval(structure)()

data_dir = 'results/' + sample_name
if not os.path.exists(data_dir):
	os.makedirs(data_dir)

# build sample and compute spectra
sample = build_stack(b, sub, main1, main2, cap)

I = reflectivity_sim(q, inst, sample)
sld = SLD_sim(q, inst, sample)
sa = spin_asymmetry(I)

# resample to N points
z = np.linspace(sld['z'].min(), sld['z'].max(), N)
re = interp1d(sld['z'], sld['Re non-mag'])(z)
im = interp1d(sld['z'], sld['Im non-mag'])(z)
mag = interp1d(sld['z'], sld['mag'])(z)
z *= 0.1
layer_bounds = 0.1*np.array(layer_bounds)

# plot reflectivity and fit
fig = Figure(figsize=(10,4.5))
canvas = FigureCanvas(fig)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
prop.set_size(18)
lprop = prop.copy()
lprop.set_size(14)

ax1.errorbar(10*df_uu['Q'].values, df_uu['R'].values, yerr=df_uu['dR'].values, lw=0, elinewidth=1, color='#316986')
s1 = ax1.scatter(10*df_uu['Q'].values, df_uu['R'].values, s=24, color='#316986', ec='none')
p1, = ax1.plot(10*q, I[0], color='#316986')

ax1.errorbar(10*df_dd['Q'].values, df_dd['R'].values, yerr=df_dd['dR'].values, lw=0, elinewidth=1, color='#C86646')
s2 = ax1.scatter(10*df_dd['Q'].values, df_dd['R'].values, s=24, color='#C86646', ec='none')
p2, = ax1.plot(10*q, I[1], color='#C86646')

ax1.set_yscale('log')
ax1.legend([s1, p1, s2, p2], ['R$^{++}$', 'Fit', 'R$^{--}$', 'Fit'], frameon=False, ncol=2, prop=lprop,
		   loc='lower left', columnspacing=1)

alpha = 0.5
ax2.plot(z, re, color=palette[0], zorder=1)
ax2.plot(z, mag, color=palette[2], zorder=2)
ax2.plot(z, im, color=palette[-1], zorder=3)
ax2.fill_between(z, re, color=palette[0], alpha=alpha - 0.2, label='NSLD', zorder=1)
ax2.fill_between(z, mag, color=palette[2], alpha=alpha, label='MSLD', zorder=2)
ax2.fill_between(z, im, color=palette[-1], alpha=alpha, label='ASLD', zorder=3)
for i in range(len(layer_bounds)):
	ax2.axvline(layer_bounds[i], linestyle='--', color='#ADABA4')

ax2.legend(prop=lprop, edgecolor='white', framealpha=1, loc='upper right')
name_pos = (layer_bounds[:-1] + layer_bounds[1:])/2.
for i, name in enumerate(layer_names[1:]):
	ax2.text(name_pos[i], 0.5, name, color='#262626', fontproperties=lprop, ha='center', va='center')

format_axis(ax1, 'Q (nm$^{-1}$)', 'Reflectivity', prop, xlims=[10*q_min, 10*q_max])
format_axis(ax2, 'z (nm)', 'SLD (10$^{-4}$/nm$^2$)', prop,
			xlims=[z.min(), z.max()], ylims=[im.min() - 0.5, re.max() + 0.5])

fig.tight_layout()
fig.subplots_adjust(wspace=0.2)
fig.savefig(data_dir + '/plot_exp_' + str(T) + 'K.pdf', dpi='figure')
