import numpy as np

fnames = ['j70s','j71s','j72s','j73s']
inds=np.arange(0.,2462.,1.)

pl=np.zeros((2,4,4))
plh=np.zeros((2,8,4))
pp=np.zeros((3,4,4))
pc=np.zeros((4,4,4))

#First: which intersection, out of 6 (or 28)
#Second: x,y
#Third: j70s-j73s

#intersections
lin=np.zeros((6,2,4))
par=np.zeros((6,2,4))
cub=np.zeros((6,2,4))
half=np.zeros((28,2,4))


#Grab data
for i in range(4):
    with np.load(fnames[i]+'intdata.npz') as d:
        half[:,:,i]=d['half']
        lin[:,:,i]=d['lin']
        par[:,:,i]=d['par']
        cub[:,:,i]=d['cub']
        pl[:,:,i]=d['pl']
        pp[:,:,i]=d['pp']
        pc[:,:,i]=d['pc']
        plh[:,:,i]=d['plh']
        
np.savez_compressed('intersects_master',half=half,lin=lin,par=par,cub=cub,pl=pl,pp=pp,pc=pc,plh=plh)