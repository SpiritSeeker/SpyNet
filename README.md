# SpyNet
A simple framework for simulating interactions between multiple neurons. 
This project is an extension of a course project for Computational Neuroscience.

## Prerequisites
The entire framework is mostly built using inbuilt python modules. 
It uses numpy for numerical optimizations.
```shell
pip install numpy
```
### GPU acceleration
Computation of dendritic voltages can be accelerated by GPUs. 
If the system has a valid installation of CUDA, install numba to make use of the GPU acceleration.
```shell
pip install numba
```
### Plotting the results (optional)
To plot the various potentials and currents, matplotlib can be used.
```shell
pip install matplotlib
```
## Structure
The framework facilitates building a network as a collection of neurons connected by synapses.
Following is a brief description. Please refer to [get_started](get_started.ipynb) for a detailed description.
### Neuron
The neuron has three components associated with it: Dendrite, Soma and Axon.
#### Dendrite
The dendrite is implemented following [Cable Theory](https://en.wikipedia.org/wiki/Cable_theory). 
I've considered the simple case of a non-branching dendrite with single synaptic input. 
A neuron can have many such dendrites and the currents due to each add up when they reach the soma.
#### Soma
The soma is a [Hodgkin-Huxley Model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) 
with input currents as the sum total of currents from all input dendrites. 
The resulting membrane potentials travel down the axon.
#### Axon
The axon follows a simplistic delay model in which the membrane potentials in the soma reach the axon terminals at delayed times.
Branching of an axon is modelled as different delays and does not affect the membrane potentials.
### Synapse
Only excitory NMDA and non-NMDA type synapses have been implemented yet. 
Each of the synapse is modelled as a change in conductance of the postsynaptic dendritic membrane.
## Acknowledgements
All the theory and models used are built using the knowledge gained by reading 
Theoretical Neuroscience by Peter Dayan and L. F. Abbott and
Biophysics of Computation by Christof Koch.
## Contributors
A big shoutout to [Chandana](https://github.com/Chandana-pvsl) for testing the framework.