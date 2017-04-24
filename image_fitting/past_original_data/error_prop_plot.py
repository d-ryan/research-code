import numpy as np
import matplotlib.pyplot as plt
import os

def comp_results(prefix):

    fs = os.listdir(prefix)
    num = len(fs)

    vars = np.zeros((18,num))
    lin = np.zeros((2,num))
    par = np.zeros((2,num))
    cub = np.zeros((2,num))

    for i,filename in enumerate(fs):
        with np.load(filename) as d:
            if i==0:
                n = d['used']
                s = d['s']
            ints = d['ints']
            cen = d['cen']
            cov = d['cov']
            param = d['param']
        for j in range(18):
            if j<4:
                w=j%2
                b=j//2
            elif j<10:
                w=(j-4)%3
                b=(j-4)//3+2
            else:
                w=(j-10)%4
                b=(j-10)//4+4
            vars[j,i]=cov[w,w,b]
        lin[:,i] = ints[0,:]-cen
        par[:,i] = ints[1,:]-cen
        cub[:,i] = ints[2,:]-cen
        
    np.savez_compressed('F:/error_prop/data_products/'+str(n)+'_'+str(s),vars=vars,lin=lin,par=par,cub=cub)