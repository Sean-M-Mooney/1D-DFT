# 1D-DFT
Simple KS 1D DFT code.

To run you must download all files and run main.py.

The config.py file contains all the input parameters and the external potential is defined in Hamiltonian.py. I've been using a Gaussian well in my testing. To see hubbard occupancy you must first create a projector by enabling write_proj in config.py and running a calculation. The wavefunction(s) of this calculation will then be written to file as the projectors. You can then enable hubbard in config.py, this will return an occupation of the hubbard subspace in the output and a perturbation to the hamiltonian if you give a non zero alpha in the input. Hubbard is limited to one subspace at the moment.
