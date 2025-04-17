# **Introduction**

This Python package serves as the frontend for calculating and building a Green's function library for synthetic seismograms and then compute the static/dynamic Coulomb Failure Change. The backend consists of Wang Rongjiang's program for calculating synthetic seismograms, including EDGRN/EDCMP and QSSP (Wang, 2003; Wang et al., 2017). The code includes two parallel modes: one using the multiprocessing library (single-node multi-process) and the other using MPI (multi-node).

Wang, R. (2003). Computation of deformation induced by earthquakes in a multi-layered elastic crust—FORTRAN programs EDGRN/EDCMP. Computers & Geosciences, 29(2), 195–207. https://doi.org/10.1016/S0098-3004(02)00111-5

Wang, R., Heimann, S., Zhang, Y., Wang, H., & Dahm, T. (2017). Complete synthetic seismograms based on a spherical self-gravitating Earth model with an atmosphere–ocean–mantle–core structure.  *Geophysical Journal International* ,  *210* (3), 1739–1764.

# Installation

1. Install the requirments. (Debian 12, Python 3.11)

```
sudo apt install default-jdk
conda create -n pycfs python=3.11
conda activate pycfs
conda install gfortran numpy scipy pandas mpi4py -c conda-forge
git clone https://github.com/Zhou-Jiangcheng/pycfs.git
cd pycfs
pip install .
```
