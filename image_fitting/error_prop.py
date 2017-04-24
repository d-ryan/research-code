import numpy as np
import matplotlib.pyplot as plt
from random import choice
import os
import sys

from numpy.random import normal
from scipy.optimize import curve_fit,root
from error_prop_plot import comp_results,plot_results

from datetime import datetime

prefix = 'F:/error_prop/cubic/'

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

def fake_spike(cen=[1000,1000],theta=np.radians(2.),n=2500,acc=0.5,excl=200,pert=None,s=0.08,sname='test'):
    '''cen : length 2 sequence giving (x,y) of true stellar center
    theta : angle b/w 0 and pi/2 giving offset of spikes from vert/horiz
    n : size of square image (integer, should be greater than cen[0] & cen[1])
    acc : between 0 and 1, fraction of image rows/columns to spikefit successfully
    excl : radius around cen to ignore in spikefitting
    pert : list of 2 sequences of (p0,p1,p2...pn) giving values of 0th,1st,2nd...nth order
           polynomial coefficients to apply as a perturbation. NOTE: p0 != 0 means
           that the spikes are NOT converging upon the star.
    '''
    
    #Establish unperturbed spike locations/formulae
    x = cen[0]
    y = cen[1]
    m = np.tan(theta)
    
    bx = y-m*x
    by = x-m*y
    
    #Set up image indices
    indsx = [float(i) for i in range(n)]
    indsy = indsx[:]

    xpix = int(x)
    ypix = int(y)
    
    #Establish exclusion region
    if xpix-excl<0:
        w = indsx.index(xpix)
        indsx = indsx[w:]
    else:
        w1 = indsx.index(xpix-excl)
        w2 = indsx.index(xpix)
        indsx = indsx[:w1]+indsx[w2:]
    if xpix+excl+1>n:
        w = indsx.index(xpix)
        indsx = indsx[:w]
    else:
        w1 = indsx.index(xpix)
        w2 = indsx.index(xpix+excl+1)
        indsx = indsx[:w1]+indsx[w2:]
    if ypix-excl<0:
        w = indsy.index(ypix)
        indsy = indsy[w:]
    else:
        w1 = indsy.index(ypix-excl)
        w2 = indsy.index(ypix)
        indsy = indsy[:w1]+indsy[w2:]
    if ypix+excl+1>n:
        w = indsy.index(ypix)
        indsy = indsy[:w]
    else:
        w1 = indsy.index(ypix)
        w2 = indsy.index(ypix+excl+1)
        indsy = indsy[:w1]+indsy[w2:]
    
    #Select successful fits at random
    num_accepted = int(acc*(n-(2*excl+1)))
    
    xs = []
    ys = []
    for i in range(num_accepted):
        k = choice(indsx)
        xs.append(k)
        indsx.remove(k)
        k = choice(indsy)
        ys.append(k)
        indsy.remove(k)
    indsx = sorted(indsx)
    indsy = sorted(indsy)
    #successful fits are built
    used = len(indsx)
    assert used ==len(indsy)
    
    #establish perturbed spike formulae
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
    
    xerr = normal(scale=s,size=len(xs))
    xmod = xs+xerr
    yerr = normal(scale=s,size=len(ys))
    ymod = ys+yerr
    
    plh,clh = curve_fit(line,indsx,ymod,sigma=s,absolute_sigma=True)
    plv,clv = curve_fit(line,indsy,xmod,sigma=s,absolute_sigma=True)
    pph,cph = curve_fit(parab,indsx,ymod,sigma=s,absolute_sigma=True)
    ppv,cpv = curve_fit(parab,indsy,xmod,sigma=s,absolute_sigma=True)
    pch,cch = curve_fit(cubic,indsx,ymod,sigma=s,absolute_sigma=True)
    pcv,ccv = curve_fit(cubic,indsy,xmod,sigma=s,absolute_sigma=True)
    cov = np.zeros((4,4,6))
    param = np.zeros((4,6))
    cov[:2,:2,0]=clh
    cov[:2,:2,1]=clv
    cov[:3,:3,2]=cph
    cov[:3,:3,3]=cpv
    cov[:,:,4]=cch
    cov[:,:,5]=ccv
    param[:2,0]=plh
    param[:2,1]=plv
    param[:3,2]=pph
    param[:3,3]=ppv
    param[:,4]=pch
    param[:,5]=pcv
    ltup = tuple(plh)+tuple(plv)
    ptup = tuple(pph)+tuple(ppv)
    ctup = tuple(pch)+tuple(pcv)

    ints = np.zeros((3,2))
    
    cenguess = (plh[0],plv[0])
    
    ints[0,:] = find_ints(crossline,cenguess,ltup)
    ints[1,:] = find_ints(crossparab,cenguess,ptup)
    ints[2,:] = find_ints(crosscubic,cenguess,ctup)
    
    #prefix = 'F:/error_prop/test/'
    sname = prefix+sname
    
    np.savez_compressed(sname,ints=ints,cen=cen,theta=theta,n=n,acc=acc,excl=excl,pert=pert,cov=cov,param=param,
        s=s,used=used)
    
    #print ints
    #print used
    #print s
    
if __name__=='__main__':
    startTime = datetime.now()
    useds = np.linspace(100,1500,15)
    ss = np.linspace(0.02,0.3,29)
    ss = np.array([0.02,0.05,0.08,0.1,0.15,0.2,0.25,0.3])
    fs = os.listdir(prefix)
    for s in ss:
        file_s = str(s).split('.')[1]
        if len(file_s)==1: file_s+='0'
        if file_s not in fs: os.mkdir(prefix+file_s)
        ffs=os.listdir(prefix+file_s)
        for uu in useds:
            u = int(uu)
            print 'Accepting',u,'with sigma',s
            #if u>600: break
            if str(u) not in ffs: os.mkdir(prefix+file_s+'/'+str(u))
            fffs = os.listdir(prefix+file_s+'/'+str(u))
            if len(fffs)!=1000:
                for i in range(1000):
                    n = int(2*u+401)
                    fake_spike(sname=file_s+'/'+str(u)+'/'+str(i),s=s,n=n,cen=[n//2,n//2],pert=[(0.,4e-5,1e-6,-1e-9),(0.,1.8e-4,0.,8e-10)])
                sname = comp_results(prefix+file_s+'/'+str(u)+'/')
            
    #plot_results(sname+'.npz')
    
    print 'Execution time: ',(datetime.now()-startTime)
    
    
    
#def simulate_errors(nx,ny,)
    
#p,c = curve_fit(line,x,y)