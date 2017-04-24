import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from emcee import PTSampler
import time
from datetime import datetime,date

#Script to apply naive and mcmc-restricted quadratic fits to the motion of fomalhaut b
#plot with "mc_motionfit_plot.py"

filename = 'fomalhaut_new.csv'

def cv(t,x0,v0):
    return x0 + v0*t

def ca(t,x0,v0,a):
    return x0 + v0*t + 0.5*a*(t**2.)

def logl(theta,t,x,y,sig_x,sig_y):
    x0 = theta[0]
    vx0 = theta[1]
    ax = theta[2]
    y0 = theta[3]
    vy0 = theta[4]
    ay = theta[5]
    
    X = ca(t,x0,vx0,ax)
    Y = ca(t,y0,vy0,ay)
    lp = -0.5*np.log(2.*np.pi)*(x.size+y.size) - np.sum(np.log(sig_x)+np.log(sig_y) + \
                    0.5*((x-X)/sig_x)**2. + 0.5*((y-Y)/sig_y)**2.)
    return lp
    
def logp(theta):

    x0 = theta[0]
    vx0 = theta[1]
    ax = theta[2]
    y0 = theta[3]
    vy0 = theta[4]
    ay = theta[5]
    
    if ax<0:
        return -np.inf
    if ay>0:
        return -np.inf
        
    return 0.

if __name__=='__main__':
    t=np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9), unpack=True).T.ravel()
    x = np.genfromtxt(filename, skip_header=1,delimiter=',',usecols=(11),unpack=True).T.ravel()/1000.
    y = np.genfromtxt(filename, skip_header=1,delimiter=',',usecols=(13),unpack=True).T.ravel()/1000.
    sig_x = np.genfromtxt(filename, skip_header=1,delimiter=',',usecols=(12),unpack=True).T.ravel()/1000.
    sig_y = np.genfromtxt(filename, skip_header=1,delimiter=',',usecols=(14),unpack=True).T.ravel()/1000.


    #Compute standard least-squares fits
    pvx,cvx = curve_fit(cv,t,x,sigma=sig_x,absolute_sigma = True)
    pvy,cvy = curve_fit(cv,t,y,sigma=sig_y,absolute_sigma = True)
    pax,cax = curve_fit(ca,t,x,sigma=sig_x,absolute_sigma = True)
    pay,cay = curve_fit(ca,t,y,sigma=sig_y,absolute_sigma = True)

    np.savez_compressed('fmh_simplefit',pvx=pvx,cvx=cvx,pvy=pvy,cvy=cvy,pax=pax,cax=cax,pay=pay,cay=cay)

    start = np.min(t)
    times = np.linspace(start-365.2425*30,start+365.2425*30,2000)

    print 'Optimal values, unaccelerated'
    print 'x: ','x0 = ',pvx[0],' v0x = ',pvx[1]
    print 'y: ','y0 = ',pvy[0],' v0y = ',pvy[1]
    print 'Accelerated'
    print 'x: ','x0 = ',pax[0],' v0x = ',pax[1],' ax = ',pax[2]
    print 'y: ','y0 = ',pay[0],' v0y = ',pay[1],' ay = ',pay[2]

    plt.errorbar(t,x,yerr=sig_x,fmt='ro',ecolor='k',elinewidth=2,capthick=2,ms=2)
    plt.errorbar(t,y,yerr=sig_y,fmt='bo',ecolor='k',elinewidth=2,capthick=2,ms=2)

    plt.plot(t,x,'ro',t,y,'bo')
    plt.plot(times,cv(times,*pvx),'r-')
    plt.plot(times,ca(times,*pax),'r--')
    plt.plot(times,cv(times,*pvy),'b-')
    plt.plot(times,ca(times,*pay),'b--')

    plt.title('Fomalhaut Naive Curve Fit')
    plt.xlabel('MJD minus 50000')
    plt.ylabel('separation')
    plt.savefig('fmh_naive.pdf')
    plt.savefig('fmh_naive.svg')

    #plt.show()

    ntemps = 20
    nwalkers = 128
    ndim = 6

    #p0 = tuple(pvx)+tuple((0,))+tuple(pvy)+tuple((0,))
    
    w0 = np.random.uniform(pvx[0]*0.9,pvx[0]*1.1, size=(ntemps,nwalkers))
    w1 = np.random.uniform(pvx[1]*0.9,pvx[1]*1.1, size=(ntemps,nwalkers))
    w2 = np.random.uniform(0,1e-3,size=(ntemps,nwalkers))
    w3 = np.random.uniform(pvy[0]*0.9,pvy[0]*1.1, size=(ntemps,nwalkers))
    w4 = np.random.uniform(pvy[1]*0.9,pvy[1]*1.1, size=(ntemps,nwalkers))
    w5 = np.random.uniform(-1e-3,0,size=(ntemps,nwalkers))
    
    p0 = np.dstack((w0,w1,w2,w3,w4,w5))
    
    savename='fmh_acc_mcmc'

    niter=30000
    nburn=10000

    time.clock()
    time.sleep(1)
    start_time=datetime.now()
    sampler=PTSampler(ntemps,nwalkers,ndim,logl,logp,loglargs=[t,x,y,sig_x,sig_y],threads=4)
    sampler.run_mcmc(p0,nburn)
    p = sampler.chain[:,:,nburn-1,:]
    np.savez_compressed(savename+'_burn',af=sampler.acceptance_fraction[0],ac=sampler.acor[0],
                chain=sampler.chain[0],lnp=sampler.lnprobability[0])
    print 'Burn in complete'
    sampler.reset()
    sampler.run_mcmc(p,niter)
    
    mx0 = np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,0]),[16,50,84])
    mvx0 = np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,1]),[16,50,84])
    max = np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,2]),[16,50,84])
    my0 = np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,3]),[16,50,84])
    mvy0 = np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,4]),[16,50,84])
    may = np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,5]),[16,50,84])
    print 'Percentiles from mcmc: '
    print 'x0 = ',mx0
    print 'y0 = ',my0
    print 'vx0 = ',mvx0
    print 'vy0 = ',mvy0
    print 'ax = ',max
    print 'ay = ',may

    np.savez_compressed(savename,af=sampler.acceptance_fraction[0],ac=sampler.acor[0],
                chain=sampler.chain[0],lnp=sampler.lnprobability[0])    
    
    print 'Total MCMC time: ', (datetime.now()-start_time)