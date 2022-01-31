## Repository for machine learning-assisted analysis of Polarized Neutron Reflectometry (PNR) measurements.

### Setup
- Clone the repository:
> `git clone https://github.com/ninarina12/ML_PNR.git`.
- Create a virtual environment with necessary dependencies:
> `conda create -n ml_pnr python=3.7.5`
> `conda install --file requirements.txt`

### Workflow
- Generate high-volume synthetic data using the GenX simulation program [1] by creating a `sim_${SAMPLE}.py` file with the appropriate sample parameters, where `${SAMPLE}` is replaced by the sample name (see examples). This script was typically executed by running `mpiexec -n num_process python sim_${SAMPLE}.py`, with `num_process` being the number of processes on which to run parallel simulations. Example parameters are based on nominal values of the target system.
- Visualize properties of the synthetic data within the `pnr_properties.ipynb` notebook.
- Train and evaluate a machine learning model within the `pnr_vae.ipynb` notebook.

#### Supporting code
- `utils` contains various utilities to assist with data import, processing, and plotting, as well as network components and architectures.

### References
[1] Björck, Matts, and Gabriella Andersson. "GenX: an extensible X-ray reflectivity refinement program utilizing differential evolution." *Journal of Applied Crystallography* 40.6 (2007): 1174-1178.