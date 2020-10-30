#!/usr/bin/env python3

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
per_group_num=int(float(parameters[9]))
dim=int(np.sqrt(N))

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
                idx2+=1
        idx1+=1

# Run Simulations
macrostate_list=np.zeros([per_group_num*3,1])
traj_all=np.zeros([(iterations+1)*per_group_num*3,N]) 
traj_all_init=np.zeros([per_group_num*3,N]) 

i=0
group_count=[0,0,0]
while not all([group>=per_group_num for group in group_count]):
    # Initialize K (Concentration Threshold) Matrix 
    k_vec=abs(np.random.normal(loc=k_mu,scale=k_sigma,size=(1,N)))

    # Initialize On/Off cell states
    if init_state==0:
        state_vec=np.array([0,1]*math.floor(N/2)+[0]*(N%2))
    if init_state==1:
        state_vec=np.zeros([1,N])

    # Start Simulation

    # Store system state at every iteration
    traj=np.zeros([iterations+1,N]) 
    traj[0,:]=state_vec

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
        traj[itr+1,:]=state_vec
    
    macro=np.mean(traj[iterations,:])
    if macro<0.25 and group_count[0]<per_group_num:
        macrostate_list[i]=macro
        traj_all[i*(iterations+1):(i*(iterations+1)+(iterations+1))]=traj
        traj_all_init[i,:]=traj[1,:]
        group_count[0]+=1
        i+=1
    elif macro>0.25 and macro<0.75 and group_count[1]<per_group_num:
        macrostate_list[i]=macro
        traj_all[i*(iterations+1):(i*(iterations+1)+(iterations+1))]=traj
        traj_all_init[i,:]=traj[1,:]
        group_count[1]+=1
        i+=1
    elif group_count[2]<per_group_num:
        macrostate_list[i]=macro
        traj_all[i*(iterations+1):(i*(iterations+1)+(iterations+1))]=traj
        traj_all_init[i,:]=traj[1,:]
        group_count[2]+=1
        i+=1
   
# PCA trajectory
pca=PCA(n_components=2)
traj_r=pca.fit(traj_all).transform(traj_all)
traj_init_r=pca.fit(traj_all_init).transform(traj_all_init)

plt.figure()
for i in range(0,per_group_num*3):
    idx=range(i*(iterations+1),(i*(iterations+1)+iterations),1)
    macro=macrostate_list[i][0]
    plt.plot(traj_r[idx,0], traj_r[idx,1],'-o', color=plt.cm.jet(macro), markersize=0, lw=1.2)

plt.figure()
for j in range(0,per_group_num*3):
    i=macro_idx[0][j]
    macro=macrostate_list_binned[i][0]
    
    ax=plt.subplot(3,per_group_num,j+1)
    ax=sbs.heatmap(np.reshape(traj_all_init[i,:],(dim,dim)))
    group_count[macro-1]+=1

plt.show()
