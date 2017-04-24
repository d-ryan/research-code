import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from orbitclass import orbit
import jdcal
from datetime import datetime,date
from emcee import PTSampler
import time

from auto_kepler import koe as kepler

#File for imports
from PZTel_params import parameters


ntemps,nwalkers,ndim,nper,nburns,nfiles,filename,savename,savetemps,prior_bounds = parameters

eps  = (np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,), unpack=True)).T.ravel()
x    = (np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(11,), unpack=True)).T.ravel()/1000
y    = (np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(13,), unpack=True)).T.ravel()/1000
sigx = (np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(12,), unpack=True)).T.ravel()/1000
sigy = (np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(14,), unpack=True)).T.ravel()/1000

eps=eps[x==x]
y = y[x==x]
sigx = sigx[x==x]
sigy = sigy[x==x]
x = x[x==x]

mass_mu = prior_bounds[6][0]
mass_sig = prior_bounds[6][1]
dist_mu = prior_bounds[7][0]
dist_sig = prior_bounds[7][1]


w0 = np.random.uniform( np.log(12),np.log(20),   size=(ntemps,nwalkers))
w1 = np.random.uniform( -0.1,0.1,    size=(ntemps,nwalkers))
w2 = np.random.uniform(np.pi/6.,np.pi/4.,  size=(ntemps,nwalkers))
w3 = np.random.uniform(np.pi/6.,np.pi/4.,    size=(ntemps,nwalkers))
w4 = np.random.uniform(-0.1,0.1,  size=(ntemps,nwalkers))
w5 = np.random.uniform( 0.1,0.2,   size=(ntemps,nwalkers))
w6 = np.random.normal(mass_mu,.2*mass_sig,size=(ntemps,nwalkers))
w7 = np.random.normal(dist_mu,.2*dist_sig,size=(ntemps,nwalkers))

p0 = np.dstack((w0,w1,w2,w3,w4,w5,w6,w7))

print 'Burning in for '+str(nper*nburns)+' steps'
print 'Sampling for '+str(nper*nfiles)+' steps'
print 'Using '+str(nwalkers)+' walkers at '+str(ntemps)+' temperatures'

def logl(theta,epochs,x,y,sigx,sigy):

    logq  = theta[0]
    tau   = theta[1]
    argp  = theta[2]
    lan   = theta[3]
    cinc  = theta[4]
    ecc   = theta[5]
    mass  = theta[6]
    dist  = theta[7]
    
    q = np.exp(logq)
    inc = np.arccos(cinc)
    
    X,Y = kepler(epochs,q,tau,argp,lan,inc,ecc,mass,dist)
    lpx = -0.5*np.log(2.*np.pi)*x.size+np.sum(-np.log(sigx)-0.5*((x-X)/sigx)**2)
    lpy = -0.5*np.log(2.*np.pi)*y.size+np.sum(-np.log(sigy)-0.5*((y-Y)/sigy)**2)
    
    return lpx+lpy
    
def logp(theta):
    
    logq  = theta[0]
    tau   = theta[1]
    argp  = theta[2]
    lan   = theta[3]
    cinc  = theta[4]
    ecc   = theta[5]
    mass  = theta[6]
    dist  = theta[7]
    
    if ecc<1:
        loga = logq-np.log(1.-ecc)
        if ((loga<np.log(1)) or (loga>np.log(500))):
            return -np.inf
    else:
        if ((logq<np.log(1)) or (logq>np.log(500))):
            return -np.inf

    if ((tau<-0.5) or (tau>0.5)):
        return -np.inf
    
    if ((argp<0) or (argp>2*np.pi)):
        return -np.inf

    if ((lan<0) or (lan>2*np.pi)):
        return -np.inf
        
    if ((cinc<-1) or (cinc>1)):
        return -np.inf
        
    if ((ecc<0) or (ecc>=1)):
        return -np.inf
        
    lnpm = -(mass-mass_mu)**2./2./mass_sig**2.
    lnpd = -(dist-dist_mu)**2./2./dist_sig**2.
    
    return lnpm+lnpd

if __name__=='__main__':
    time.clock()
    time.sleep(1)
    startTime = datetime.now()
    sampler=PTSampler(ntemps, nwalkers, ndim, logl, logp ,loglargs=[eps,x,y,sigx,sigy],threads=4)
    st = savetemps
    p = p0
    for index in range(nburns):
        sampler.run_mcmc(p,nper)
        p = sampler.chain[:,:,nper-1,:]
        np.savez_compressed(savename+'_burn_'+str(index),af=sampler.acceptance_fraction[st],ac=sampler.acor[st],
                chain=sampler.chain[st],lnp=sampler.lnprobability[st])
        print 'Finished burn '+str(index+1)+' of '+str(nburns)
        sampler.reset()
    for index in range(nfiles):
        sampler.run_mcmc(p,nper)
        p = sampler.chain[:,:,nper-1,:]
        if index == nfiles-1:
            print 'Final sampling run parameter ranges:'
            PERI = np.exp(np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,0]),[16,50,84]))
            LAN = np.degrees(np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,3]),[16,50,84]))
            INC = np.degrees(np.arccos(np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,4]),[16,50,84])))
            ECC = np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,5]),[16,50,84])
            
            periods = np.sqrt((sampler.flatchain[0,:,0]/(1.-sampler.flatchain[0,:,5]))**3./sampler.flatchain[0,:,6])
            
            PER = np.percentile(np.ndarray.flatten(periods),[16,50,84])
            print 'PERI = ',PERI,' AU'
            print 'LAN = ',LAN,' deg'
            print 'inc = ', INC,' deg'
            print 'ecc = ',ECC
            print 'Per = ',PER,' yr'
        np.savez_compressed(savename+str(index),af=sampler.acceptance_fraction[st],ac=sampler.acor[st],
                chain=sampler.chain[st],lnp=sampler.lnprobability[st])
        print 'Saved samples, file '+str(index+1)+' of '+str(nfiles)
        sampler.reset()
    print 'Total Execution Time: ',(datetime.now()-startTime)
    