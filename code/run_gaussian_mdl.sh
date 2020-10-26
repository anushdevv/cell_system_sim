#!/bin/sh

N=400 # Number of cells on the lattice
k_mu=880 # Mean of concentration threshold distribution 
k_sigma=300 # Standard Deviation of concentration threshold distribution 
lambd=10 # Signaling strength (regulates rate of decay with distance)
r0=1 # Distance between adjacent cells on lattice
s_on=10 # ON state concentration
iterations=100 
simulations=400

python3 ./gaussian_mdl.py $N $k_mu $k_sigma $lambd $r0 $s_on $iterations $simulations
