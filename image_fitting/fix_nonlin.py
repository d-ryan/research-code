import numpy as np
import matplotlib.pyplot as plt
from random import choice
import os
import sys

from numpy.random import normal
from scipy.optimize import curve_fit,root
from error_prop_plot import comp_results,plot_results

prefix = 'F:/error_prop/parab/'

#fname = 'F:/error_prop/cubic/02/1400/0.npz'

def find_ints(fn,guess,params):
    #pdb.set_trace()
    soln=root(fn,guess,args=params)
    if soln.success:
        return soln.x[0],soln.x[1]
    else:
        return -1,-1

def line(x,b,m):
    y = m*x + b
    return y
    
def parab(x,c,b,a):
    y = a*(x**2) + b*x + c
    return y

def cubic(x,d,c,b,a):
    y = a*(x**3) + b*(x**2) + c*x + d
    return y
    
def crossparab(xarr,c,b,a,f,e,d):
    x=xarr[0]
    y=xarr[1]
    fy=parab(x,c,b,a)
    fx=parab(y,f,e,d)
    return np.array([x-fx,y-fy])
    
def crossline(xarr,b,m,c,n):
    x=xarr[0]
    y=xarr[1]
    fy=line(x,b,m)
    fx=line(y,c,n)
    return np.array([x-fx,y-fy])
    
def crosscubic(xarr,d,c,b,a,h,g,f,e):
    x=xarr[0]
    y=xarr[1]
    fy=cubic(x,d,c,b,a)
    fx=cubic(y,h,g,f,e)
    return np.array([x-fx,y-fy])



def fixer(dirpath):
    for fname in os.listdir(dirpath):
        with np.load(dirpath+'/'+fname) as d:
            ints = d['ints']
            cen = d['cen']
            theta = d['theta']
            n = d['n']
            acc = d['acc']
            excl = d['excl']
            pert = d['pert']
            cov = d['cov']
            param = d['param']
            s=d['s']
            used=d['used']
            
            x = cen[0]
            y = cen[1]
            m = np.tan(theta)
            
            bx = y-m*x
            by = x-m*y
            
            indsx = [float(i) for i in range(n)]
            indsy = indsx[:]
            if pert is not None:
                #Then it should be a sequence of (p0,p1,p2,p3) giving the values of
                #the 0th, 1st, 2nd, and 3rd-order polynomial coefficients to apply as a
                #perturbation
                xp = pert[0]
                yp = pert[1]
                xorder = len(xp)
                yorder = len(yp)
                xp = list(xp)
                yp = list(yp)
                xp[0]+=bx
                yp[0]+=by
                if xorder>1:
                    xp[1]+=m
                else:
                    xp.append(m)
                if yorder>1:
                    yp[1]+=m
                else:
                    xp.append(m)
                while len(xp)<4:
                    xp.append(0.)
                while len(yp)<4:
                    yp.append(0.)
            else:
                xp = [bx,m,0.,0.]
                yp = [by,m,0.,0.]
                
            ys = [cubic(i,*xp) for i in indsx]
            xs = [cubic(i,*yp) for i in indsy]
            
            ys = np.array(ys)
            xs = np.array(xs)
            
            cenguess = cen
            
            ctup = tuple(xp)+tuple(yp)
            
            true_int = find_ints(crosscubic,cenguess,ctup)
            
        np.savez_compressed(dirpath+'/'+fname[:-4],ints=ints,cen=true_int,theta=theta,n=n,acc=acc,excl=excl,pert=pert,cov=cov,param=param,
                s=s,used=used)
                
if __name__=='__main__':
    fs = os.listdir(prefix)
    for file in fs:
        ffs = os.listdir(prefix+'/'+file)
        for ff in ffs:
            dirpath = prefix+'/'+file+'/'+ff
            fixer(dirpath)
            sname = comp_results(prefix+'/'+file+'/'+ff+'/')
            print 'Fixed',file,ff