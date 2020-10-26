#!/bin/sh

N=400
k_mu=880
k_sigma=300
lambd=10
r0=1
s_on=10
iterations=100
simulations=400

python3 ./gaussian_mdl.py $N $k_mu $k_sigma $lambd $r0 $s_on $iterations $simulations
