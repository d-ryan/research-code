##IRRELEVANT, use intersect_plot.py instead


import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


'''MAKE THIS GO, THIS IS THE LAST STEP'''
fnames = ['j70s','j71s','j72s','j73s']
inds=np.arange(0.,2462.,1.)

def line(x,m,b):
    return m*x+b

def parab(x,a,b,c):
    return a*x**2 + b*x + c

def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

'''NOT DONE'''
for i in range(4):
    if i==0: ##Need to save: half-spike imgs for j72s thru j73s.
        continue #And full-spike lin, par, cub imgs for ALL.
    elif i==1:
        continue
    #Also should really make a master file of ALL intersects for ALL images.
    with np.load(fnames[i]+'intdata.npz') as d:
        half=d['half']
        lin=d['lin']
        par=d['par']
        cub=d['cub']
        pl=d['pl']
        pp=d['pp']
        pc=d['pc']
        plh=d['plh']
    
    with np.load('rot'+fnames[i]+'.fits.npz') as d:
        pos=d['pos']
    
    if i==0:
        pos[pos>250]=250
        pos[pos<167]=167
    if i==1:
        pos[pos<237]=237
        pos[pos>350]=350
    if i==2:
        pos[pos>250]=250
        pos[pos<161]=161
    if i==3:
        pos[pos>250]=250
        pos[pos<150]=150
    
    #Half lines
    plt.imshow(pos,origin='lower')
    for j in range(8):
        plt.plot(inds,line(inds,*plh[:,j]))
    plt.plot(half[:,0],half[:,1],'bo')
    plt.xlim([0,2462])
    plt.ylim([0,2462])
    plt.show()
    
    for j in range(8):
        plt.plot(inds,line(inds,*plh[:,j]))
    plt.plot(half[:,0],half[:,1],'bo')
    plt.xlim([0,2462])
    plt.ylim([0,2462])
    plt.show()
    
    
    #Full lines
    
    #Quadratics
    
    #Cubics
    
    #Everything
    
    #plt.savefig(1hewuibpub)