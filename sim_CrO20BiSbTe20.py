import numpy as np
from utils.utils_format import format_parameters, format_metadata, make_dirs, clean_dirs
from utils.utils_simulate import u, bc, run

# data properties
num_samples = 100000                                    # number of examples to generate
N = 256                                                 # number of q points for reflectivity plot
q_min = 0.01
q_max = 0.172
q = np.linspace(q_min, q_max, N)                        # q points at which to sample reflectivity (A^-1)
t_min = 2.												# thickness resolution (<= wavelen)

# instrument properties
wavelen = 5.35                                          # neutron wavelength (A)
res = 0.5*(0.001 + 0.0001)								# instrument resolution
dres = 0.5*(0.001 - 0.0001)							    # uncertainty in instrument resolution
log_bkg = -6											# instrument background (log10)
dlog_bkg = 2											# uncertainty in instrument background (log10)

# sample properties
def init_sample():
	sample_name = 'CrO20_BiSbTe20'                          # name of experiment sample
	r = 0													# max ratio m_prox/m_FM (emu/cm^3)

	layer_names = [r'$\alpha-Al_2O_3$', '$Cr_2O_3$', '$Cr_2O_3$\ninterface', '$(Bi,Sb)_2Te_3$\ninterface', '$(Bi,Sb)_2Te_3$', 'Te', '$TeO_2$']
	rho = [3.95, 5.22, 5.22, 0.2*7.7 + 0.8*6.5, 0.2*7.7 + 0.8*6.5, 6.24, 5.67]					  # nominal densities (g/cm^3)
	M = [101.96, 151.99, 151.99, 0.2*800.76 + 0.8*626.32, 0.2*800.76 + 0.8*626.32, 127.6, 159.6]  # molar masses (g/mol)

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
	b_main1 = bc.Cr*2+bc.O*3
	dens_main1 = np.array([[rho[1]/u/M[1]]])
	ddens_main1 = 0.2*dens_main1
	v_main1 = np.array([[200., 1., 0], [10., 5., 0.4]])
	dv_main1 = np.array([[20., 0., 0], [10., 5., 0.4]])

	# main2 (2 layers)
	b_main2 = bc.Bi*0.4+bc.Sb*1.6+bc.Te*3
	dens_main2 = np.array([[rho[3]/u/M[3]]])
	ddens_main2 = 0.3*dens_main2
	v_main2 = np.array([[10., 1., 0.4], [180., 20., 0.]])
	dv_main2 = np.array([[10., 0., 0.4], [20., 10., 0.]])

	# cap
	b_cap = [bc.Te, bc.Te+bc.O*2]
	dens_cap = np.array([[rho[5]/u/M[5]], [rho[6]/u/M[6]]])
	ddens_cap = np.array([[0.5], [0.5]])*dens_cap
	v_cap = np.array([[90., 30., 0], [25., 20., 0]])
	dv_cap = np.array([[20., 10., 0], [10., 10., 0]])

	# interfacial FM layer(s)
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
