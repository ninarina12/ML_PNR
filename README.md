## Executable scripts
pnr_generate.py 		generate data
pnr_properties.ipynb 	plot statistics of synthetic data
pnr_vae.ipynb 			execute model training and analysis

## Helper scripts
pnr_models.py 		all network components and architectures are stored here
pnr_utils.py 		various utilities to assist with data import, processing, and plotting
plot_imports.py 	some typical imports for plotting

####################################################################################################
## Examples

### Generating data using MPI:
nice -n 19 mpiexec -n 25 python pnr_generate.py

### Copying results to local computer while excluding big data files:
rsync -rav -e 'ssh -p 2866' --exclude='data' --exclude='model.torch' user@atom.mit.edu:~/data1/pnr/final/results /path/on/local/machine