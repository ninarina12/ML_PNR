Repository for machine learning-assisted analysis of Polarized Neutron Reflectometry (PNR) data.

### Setup
- Clone the repository with `git clone https://github.com/ninarina12/ML_PNR.git`.
- Create a conda environment with necessary dependencies using `conda create --name myenv --file env_file.txt`.

### Workflow
- Generate synthetic data by updating `pnr_generate.py` with the appropriate sample parameters and executing `mpiexec -n num_process python pnr_generate.py`, with `num_process` being the number of processes on which to run parallel simulations. Example parameters are based on nominal values of measurements in the `experiments` directory.
- Visualize properties of the synthetic data directly within the `pnr_properties.ipynb` notebook.
- Train and evaluate a machine learning model within the `pnr_vae.ipynb` notebook.

#### Helper scripts
- `pnr_models.py` contains all network components and architectures.
- `pnr_utils.py` contains various utilities to assist with data import, processing, and plotting.
- `plot_imports.py` contains some typical imports for plotting.

#### Other
- `plot_exp.py` can be used to fit and plot experimental data only.