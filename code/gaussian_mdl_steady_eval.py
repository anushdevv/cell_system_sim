#!/usr/bin/env python3

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as sp
import seaborn as sbs

# Parameters
parameters=sys.argv

N=int(float(parameters[1]))
k_mu=float(parameters[2])
k_sigma=float(parameters[3])
lambd=float(parameters[4])
r0=float(parameters[5])
s_on=float(parameters[6])
iterations=int(float(parameters[7]))
init_state=int(float(parameters[8]))
simulations=int(float(parameters[9]))
show_steady=int(float(parameters[10]))
num_rads=int(float(parameters[11]))
dim=int(np.sqrt(N))
    
# Index of centre cell
if dim%2==1:
    idxc=int(math.ceil(0.5*N))-1
else:
    idxc=int(0.5*(N+dim))-1
thetas=np.zeros(N)

# All pairwise distances on Lattice
dist_mat=np.zeros([N,N])
idx1=0
for x1 in range(0,dim):
    for y1 in range(0,dim):
        idx2=0
        for x2 in range(0,dim):
            for y2 in range(0,dim):
                dist=np.sqrt((r0*(x2-x1))**2+(r0*(y2-y1))**2)
                dist_mat[idx1,idx2]=dist
                
                if idx1==idxc:
                    x=r0*(x2-x1)
                    y=r0*(y2-y1)
                    theta=np.arctan(y/float(x+0.000000001))
                    thetas[idx2]=theta
                    
                idx2+=1
        idx1+=1

# Run Simulations
macrostate_list=np.zeros([simulations,1])
for i in range(0,simulations):
    
    # Initialize K (Concentration Threshold) Matrix 
    k_vec=abs(np.random.normal(loc=k_mu,scale=k_sigma,size=(1,N)))

    # Initialize On/Off cell states
    if init_state==0:
        state_vec=np.array([0,1]*math.floor(N/2)+[0]*(N%2))
    elif init_state==1:
        state_vec=np.zeros([1,N])

    # Start Simulation
    
    # Store system state at every iteration
    traj=np.zeros([N,iterations+1]) 
    traj[:,0]=state_vec
    
    for itr in range(0,iterations):
        
        # Compute concentration at surface of all N cells
        tmp_conc=state_vec*s_on
        tmp_conc[np.where(tmp_conc==0)]=1
        
        # Compute concentration sensed by cell i
        tmp_conc_vec=np.zeros(N)
        for cell in range(0,N):
            tmp_dist=dist_mat[cell,:]
            decay=np.exp(-(1/lambd)*tmp_dist)
            conc=np.matmul(decay,np.transpose(tmp_conc))
            tmp_conc_vec[cell]=conc
        
        # Update system state
        state_vec=(tmp_conc_vec>k_vec)*1
        traj[:,itr+1]=state_vec
    
    steady=traj[:,iterations]
    macrostate_list[i]=np.mean(steady)
    
    if show_steady==1:
        plt.figure()
        sbs.heatmap(np.reshape(steady,(dim,dim)))
        
    # Evaluation Function
    plt.figure()
    srt_idx=np.argsort(thetas)
    dist_mat_c=dist_mat[idxc,srt_idx]
    steady_srt=steady[srt_idx]

    rads=np.linspace(0,max(dist_mat_c),num_rads+1)
    for idx in range(0,len(rads)-1):
        rad_idx=np.array([idx2 for idx2 in range(0,len(dist_mat_c)) if (dist_mat_c[idx2]>rads[idx] and dist_mat_c[idx2]<=rads[idx+1])])
        if len(rad_idx>0):
            rad_state=steady_srt[rad_idx]
            plt.subplot(1,len(rads)-1,idx+1)
            plt.plot(list(range(1,len(rad_state)+1)),rad_state,lw=0.5)
            
plt.show()
