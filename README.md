## Repository for machine learning-assisted analysis of Polarized Neutron Reflectometry (PNR) measurements.

### Setup
- Clone the repository with `git clone https://github.com/ninarina12/ML_PNR.git`.
- Create a conda environment with necessary dependencies using `conda create --name myenv --file env_file.txt`.

### Workflow
- Generate high-volume synthetic data using the GenX simulation program [1] by updating `pnr_generate.py` with the appropriate sample parameters and running `mpiexec -n num_process python pnr_generate.py`, with `num_process` being the number of processes on which to run parallel simulations. Example parameters are based on nominal values of the target system.
- Visualize properties of the synthetic data directly within the `pnr_properties.ipynb` notebook.
- Train and evaluate a machine learning model within the `pnr_vae.ipynb` notebook.

#### Supporting code
- `pnr_models.py` contains all network components and architectures.
- `pnr_utils.py` contains various utilities to assist with data import, processing, and plotting.
- `plot_imports.py` contains some typical imports for plotting.
- `plot_sld.py` allows for plotting the SLD profile from a set of parameter predictions.

### References
[1] Björck, Matts, and Gabriella Andersson. "GenX: an extensible X-ray reflectivity refinement program utilizing differential evolution." *Journal of Applied Crystallography* 40.6 (2007): 1174-1178.