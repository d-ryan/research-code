import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv

fnames = ['j70s','j71s','j72s','j73s']
inds=np.arange(0.,2462.,1.)

with np.load('intersects_master.npz') as d:
    
    #First: which intersection, out of 6 (or 28)
    #Second: x,y
    #Third: j70s-j73s
    half=d['half']
    lin=d['lin']
    par=d['par']
    cub=d['cub']

aves = np.zeros((4,2,4))
    
z=0    
for j in range(3):
    for k in range(j+1,4):
        if (j<2 and k<2):
            z+=1
            continue
        elif (j>=2 and k>=2):
            z+=1
            continue
        else:
            aves[0,:,:]+=lin[z,:,:]/4.
            aves[1,:,:]+=par[z,:,:]/4.
            aves[2,:,:]+=cub[z,:,:]/4.
            z+=1

z=0
for j in range(7):
    for k in range(j+1,8):
        if (j<4 and k<4):
            z+=1
            continue
        elif (j>=4 and k>=4):
            z+=1
            continue
        else:
            aves[3,:,:]+=half[z,:,:]/16.
            
np.savez_compressed('ave_ints',aves=aves)

methods=['linear','parabolic','cubic','half']

with open('ave_ints.csv','wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image','Method','x','y'])
    for i in range(4):
        for j in range(4):
            writer.writerow([fnames[i],methods[j]]+list(aves[j,:,i]))