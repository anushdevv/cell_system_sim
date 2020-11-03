#!/bin/sh

######### Parameter Set 1 #########

N=400 # Number of cells on the lattice
k_mu=880 # Mean of concentration threshold distribution 
k_sigma=300 # Standard Deviation of concentration threshold distribution 
lambd=10 # Signaling strength (regulates rate of decay with distance)
r0=1 # Distance between adjacent cells on lattice
s_on=10 # ON state concentration
iterations=50
init_state=0
simulations=30
per_group_num=30 # Run simulations until per_group_num of each steady state type is found
show_steady=0 # 1: plot steady state for every simulation, 0: Do nothing

# Compute proportion of ON cells at steady state
#python3 ./gaussian_mdl.py $N $k_mu $k_sigma $lambd $r0 $s_on $iterations $init_state $simulations $show_steady

# Analyze trajectories
#python3 ./gaussian_mdl_traj_analysis.py $N $k_mu $k_sigma $lambd $r0 $s_on $iterations $init_state $per_group_num

######### Parameter Set 2 #########

N=10000 # Number of cells on the lattice
k_mu=5.5 # Mean of concentration threshold distribution 
k_sigma=1.5 # Standard Deviation of concentration threshold distribution 
lambd=0.5 # Signaling strength (regulates rate of decay with distance)
r0=1 # Distance between adjacent cells on lattice
s_on=10 # ON state concentration
iterations=100
init_state=1
simulations=1
show_steady=1 # 1: plot steady state for every simulation, 0: Do nothing
num_rads=3 # Number of radii to sample
fft_thresh=1

# Analyze/visualize steady states
#python3 ./gaussian_mdl.py $N $k_mu $k_sigma $lambd $r0 $s_on $iterations $init_state $simulations $show_steady

# Evaluate function for steady states
#python3 ./gaussian_mdl_steady_eval.py $N $k_mu $k_sigma $lambd $r0 $s_on $iterations $init_state $simulations $show_steady $num_rads

# Evaluate function 2 for steady states
python3 ./gaussian_mdl_steady_eval2.py $N $k_mu $k_sigma $lambd $r0 $s_on $iterations $init_state $simulations $show_steady $num_rads $fft_thresh
