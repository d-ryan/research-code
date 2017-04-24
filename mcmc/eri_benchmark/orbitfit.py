import numpy as np
from emcee import PTSampler
import time
from koe import koe
from eri_constants import constants
from datetime import datetime,date
import jdcal
import pdb
from orbitclass import orbit
import matplotlib.pyplot as plt
import pickle as pickle 

'''when switching stars, change:
--priors
--save filename
--walker locations
--constants file import, here AND in koe
--chain thinning, if desired
'''
tau_max=jdcal.gcal2jd(date.today().year,date.today().month,date.today().day)[1]
years=100
tau_min=tau_max-years*365.2425

savename='eri_benchmark'

'''
eritest2 - linear prior, short
eritest3 - flat prior, short
eritest4 - flat prior, long
'''
parallax,distance,mstar,kgauss, AU, DAY = constants()
def logl(theta,eps,xy,sig):
    loga = theta[0]
    tau  = theta[1]
    argp = theta[2]
    lan  = theta[3]
    cinc = theta[4]
    ecc  = theta[5]

    
    a = np.exp(loga)
    inc = np.arccos(cinc)
    
    XY=koe(eps,a,tau,argp,lan,inc,ecc)
    
    lpx = -0.5*np.log(2*np.pi) * xy.size + \
                   np.sum( -np.log(sig)- 0.5*( (xy-XY)/sig)**2)
    
    return lpx

def logp(theta): #FOR 51 eri

    loga  = theta[0]
    tau   = theta[1]
    argp  = theta[2]
    lan   = theta[3]
    cinc  = theta[4]
    ecc   = theta[5]

    # include priors here

    if (loga < np.log(1) or loga > np.log(100)): 
        return -np.inf

    if (tau < -0.5 or tau > 0.5):
        return -np.inf
 
    if (argp < 0 or argp > 2*np.pi):
        return -np.inf

    if (lan < 0 or lan > 2*np.pi):
        return -np.inf

    if (cinc < 0 or cinc > 1): 
        return -np.inf

    if (ecc < 0 or ecc >= 0.95724):
    #if (ecc <0.4 or ecc >= 1.0):
        return -np.inf
   
    # otherwise ... 
    #return 0.
    return np.log(-2.1826563*ecc+2.0893331)
print 'likelihoods & priors built!'
######################################################## 

ntemps   = 20 #Number of temperatures for the parallel tempering
nwalkers = 128 #Number of walkers
ndim     = 6 #Number of dimensions in the orbit fit

filename='51eri.csv' #Beta Pic
#filename='fomalhaut_new.csv' #Fomalhaut
eps=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,10), unpack=True)).T.ravel()
xy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(11,13), unpack=True)).T.ravel()/1000
sigs=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(12,14), unpack=True)).T.ravel()/1000
eps=eps[xy == xy]
sigs=sigs[xy == xy]
xy=xy[xy == xy]

'''
#FOR FOMALHAUT
w0 = np.random.uniform( 4.0,7.0,    size=(ntemps,nwalkers))
w1 = np.random.uniform( -0.25,0.25,    size=(ntemps,nwalkers))
w2 = np.random.uniform(-np.pi/4.,3.*np.pi/4.,  size=(ntemps,nwalkers))
w3 = np.random.uniform( np.pi/3,np.pi,    size=(ntemps,nwalkers))
w4 = np.random.uniform(-0.3,0.3,  size=(ntemps,nwalkers))
w5 = np.random.uniform( 0.785,0.825,   size=(ntemps,nwalkers))
'''
'''
#FOR BETA PIC
w0 = np.random.uniform( 2.0,2.5,    size=(ntemps,nwalkers))
w1 = np.random.uniform( -0.25,0.25,    size=(ntemps,nwalkers))
w2 = np.random.uniform(-3*np.pi/4.,-np.pi/4.,  size=(ntemps,nwalkers))
w3 = np.random.uniform( 0.8,1.2,    size=(ntemps,nwalkers))
w4 = np.random.uniform(-0.05,0.05,  size=(ntemps,nwalkers))
w5 = np.random.uniform( 0.05,0.2,   size=(ntemps,nwalkers))
'''
#'''
#for 51 eri
w0 = np.random.uniform( 2.3,3,    size=(ntemps,nwalkers))
w1 = np.random.uniform( -0.1,0.1,    size=(ntemps,nwalkers))
w2 = np.random.uniform(np.pi/4.,3*np.pi/4.,  size=(ntemps,nwalkers))
w3 = np.random.uniform( 0.8,1.2,    size=(ntemps,nwalkers))
w4 = np.random.uniform(-0.1,0.1,  size=(ntemps,nwalkers))
w5 = np.random.uniform( 0.1,0.2,   size=(ntemps,nwalkers))
#'''

p0 = np.dstack((w0,w1,w2,w3,w4,w5))

#Numnber of iterations (after the burn-in)
niter = 10000
num_files=1 #I trust YOU, user, to make sure niter / num_files is remainderless

nperfile=niter/num_files

#Burn-in for 1/5 of the number of iterations
nburn = 10000
print 'Made it this far, waiting 5 secs'
if __name__=='__main__':
    time.clock()
    time.sleep(1)
    startTime = datetime.now()
    sampler=PTSampler(ntemps, nwalkers, ndim, logl, logp ,loglargs=[eps,xy,sigs],threads=4)
    sampler.run_mcmc(p0,nburn)
    for index in range(num_files):
        if index == 0:
            p = sampler.chain[:,:,nburn-1,:]
            np.savez_compressed(savename+'_burn_'+str(index),af=sampler.acceptance_fraction[0],ac=sampler.acor[0],
                chain=sampler.chain[0],lnp=sampler.lnprobability[0])
            print 'Burn in complete'
        else:
            p = sampler.chain[:,:,nperfile-1,:]
        sampler.reset()
        #print 'Burn in complete'
        sampler.run_mcmc(p,nperfile)
        #print 'orbit fitting complete'

        if index == num_files-1:
            SMA = np.exp(np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,0]),[16,50,84]))
            LAN = np.degrees(np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,3]),[16,50,84]))
            INC = np.degrees(np.arccos(np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,4]),[16,50,84])))
            ECC = np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,5]),[16,50,84])
            PER = 2*np.pi/(kgauss*np.sqrt(mstar)*(SMA)**(-1.5) ) /365.25
            print 'SMA = ',SMA,'; AU'
            print 'LAN = ',LAN,' deg'
            print 'inc = ', INC,' deg'
            print 'ecc = ',ECC
            print 'Per = ',PER,' yr'


        np.savez_compressed(savename+'_'+str(index),af=sampler.acceptance_fraction[0],ac=sampler.acor[0],
                            chain=sampler.chain[0],lnp=sampler.lnprobability[0])

    with open('time.txt','w') as f:
        f.write('Total Execution time: '+str(datetime.now()-startTime))
    #print 'Total Execution Time:', (datetime.now()-startTime)
    #pdb.set_trace()