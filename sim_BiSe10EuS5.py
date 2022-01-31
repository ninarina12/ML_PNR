import numpy as np
from utils.utils_format import format_parameters, format_metadata, make_dirs, clean_dirs
from utils.utils_simulate import u, bc, run

# data properties
num_samples = 200000                                    # number of examples to generate
N = 256                                                 # number of q points for reflectivity plot
q_min = 0.01
q_max = 0.13
q = np.linspace(q_min, q_max, N)                        # q points at which to sample reflectivity (A^-1)
t_min = 2.												# thickness resolution (<= wavelen)

# instrument properties
wavelen = 4.75                                          # neutron wavelength (A)
res = 0.5*(0.001 + 0.0001)								# instrument resolution
dres = 0.5*(0.001 - 0.0001)							    # uncertainty in instrument resolution
log_bkg = -6											# instrument background (log10)
dlog_bkg = 2											# uncertainty in instrument background (log10)

# sample properties
def init_sample():
	sample_name = 'BiSe10_EuS5'                             # name of experiment sample
	r = 0.5													# max ratio m_prox/m_FM (emu/cm^3)
	
	layer_names = [r'$\alpha-Al_2O_3$', '$Bi_2Se_3$', '$Bi_2Se_3$\ninterface', 'EuS', '$a-Al_2O_3$']
	rho = [3.95, 6.82, 6.82, 5.75, 2.75]					# nominal densities (g/cm^3)
	M = [101.96, 654.8, 654.8, 184.03, 101.96]				# molar masses (g/mol)

	# variables (v) = [d, s, magn]
	# b = neutron scattering length
	# dens = density (g/cm^3)/u/molar mass of compound (g/mol) = density (fu/A^3)
	# d = layer thickness (A)
	# s = roughness of layer top surface (A)
	# magn = magnetization (uB/fu)

	# substrate
	b_sub = bc.Al*2+bc.O*3
	dens_sub = np.array([[rho[0]/u/M[0]]])
	ddens_sub = 0.05*dens_sub
	v_sub = np.array([[0, 3., 0]])
	dv_sub = np.array([[0, 0., 0]])

	# main1 (2 layers)
	b_main1 = bc.Bi*2+bc.Se*3
	dens_main1 = np.array([[rho[1]/u/M[1]]])
	ddens_main1 = 0.2*dens_main1
	v_main1 = np.array([[100., 1., 0], [15., 5., 1.]])
	dv_main1 = np.array([[20., 0., 0], [15., 5., 1.]])

	# main2
	b_main2 = bc.Eu+bc.S
	dens_main2 = np.array([[rho[3]/u/M[3]]])
	ddens_main2 = 0.2*dens_main2
	v_main2 = np.array([[60., 8., 4.]])
	dv_main2 = np.array([[20., 6., 4.]])

	# cap
	b_cap = [bc.Al*2+bc.O*3]
	dens_cap = np.array([[rho[4]/u/M[4]]])
	ddens_cap = np.array([[0.5]])*dens_cap
	v_cap = np.array([[100., 10., 0]])
	dv_cap = np.array([[30., 6., 0]])

	# interfacial FM layer(s)
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
	
	sample_props = sample_name, layer_names, rho, M, stack, dstack, prox
	params = b, dens, ddens, v, dv, prox, r

	return sample_props, header, params, stack, dstack

if __name__ == "__main__":
	# build sample
	sample_props, header, params, stack, dstack = init_sample()

	# create metadata
	data_props = N, q_min, q_max, t_min
	inst_props = wavelen, res, dres, log_bkg, dlog_bkg
	meta = format_metadata(data_props, inst_props, sample_props, params[-1])

	# make directories
	data_dir, data_files, comm, rank, start, stop, size, t0 = make_dirs(num_samples, sample_props[0], meta)

	# run simulations
	run(data_files, start, stop, q, t_min, *inst_props, *params)

	# clean up files
	clean_dirs(data_dir, data_files, comm, rank, size, t0, header + ['I_res', 'I_bkg'])
