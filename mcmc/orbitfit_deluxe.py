#Goal:
#Produce single-file (apt from csv) orbit fitting code
#With OPTIONS to include uncertainty in mass, distance, or both
#Taus should be in MJD, not relative phase
#Other considerations:
# --Consequences of including a in " rather than in AU?
# --Consequences of fitting an anomaly (either of a specific epoch, or today) rather than tau
# --Consequences of fitting period rather than tau


#Notes: a,e,i,\omega,\Omega : describe size and shape of orbit.

#You should need EXACTLY EIGHT parameters to fully describe the two-body problem.
#Need BOTH MASSES. (2)
#Need 3 Euler angles (3)
#Need size & shape (2)
#Need phase (1).


#Two possible approaches:
#1) Physics: Current approach.
#       Puts observations (x,y)=f(t) into physical units (AU) from arcseconds (needs distance).
#       Guesses Keplerian elements, sees what orbit gets generated & how well it matches obs (uses mass
#           but does not need to).
#       
#2) Geometry: Hypothetical approach.
#       Keeps a in arcseconds. Guesses remaining geometrical elements (e & Euler angles) & period.
#       Checks these against data for fit.



import numpy as np
from emcee import PTSampler
import time
from datetime import datetime,date
import jdcal
import pdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from koe import koe
from fmh_constants import constants
from orbitclass import orbit

tau_max = jdcal.gcal2jd(date.today().year,date.today().month,date.today().day)[1]
years = 100
tau_min = tau_max - years*365.2425

savename = 'deluxe_test'

parallax,distance,mstar,kgauss,AU,day = constants()

def constants():

def koe(epochs,a,tau,argp,lan,inc,ecc):

    from etest import ema
    import numpy as np 
    
    from fmh_constants import constants

    # Epochs in MJD
    # 
    # date twice but x & y are computed.  
    # The data are returned so that the values in the array
    # alternate x and y pairs.
    #
    # Stellar properties for beta Pic

    parallax,distance,mstar,kgauss,AU,DAY = constants()
                                       
    ndate = len(epochs)

    # Keplerian Elements 
    #
    # epochs      dates in modified JD [day]
    # a           semimajor axis [au]
    # tau         epoch of peri in units of the orbital period
    # argp        argument of peri [radians]
    # lan         longitude of ascending node [radians]
    # inc         inclination [radians]
    # ecc         eccentricity 
    #
    # Derived quantities 
    # manom   --- mean anomaly
    # eccanom --- eccentric anomaly
    # truan   --- true anomaly
    # theta   --- longitude 
    # radius  --- star-planet separation
    n = kgauss*np.sqrt(mstar)*(a)**(-1.5)  # compute mean motion in rad/day
    # ---------------------------------------
    # Compute the anomalies (all in radians)
    #
    # manom = n * (epochs - tau) # mean anomaly 
    manom = n*epochs[0::2] - 2*np.pi*tau  # Mean anomaly w/tau in units of period
    eccanom = np.array([])
    for man in manom:
        #print man #DEBUGGING
        eccanom = np.append(eccanom, ema(man%(2*np.pi), ecc))
    # ---------------------------------------
    # compute the true  anomaly and the radius
    #
    # Elliptical orbit only
    truan = 2.*np.arctan(np.sqrt( (1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eccanom) )
    theta = truan + argp
    radius = a * (1.0 - ecc * np.cos(eccanom))
    # ---------------------------------------
    # Compute the vector components in 
    # ecliptic cartesian coordinates (normal convention Green Eq. 7.7)
    # standard form
    #
    #xp = radius *(np.cos(theta)*np.cos(lan) - np.sin(theta)*np.sin(lan)*np.cos(inc))
    #yp = radius *(np.cos(theta)*np.sin(lan) + np.sin(theta)*np.cos(lan)*np.cos(inc))
    #
    # write in terms of \Omega +/- omega --- see Mathematica notebook trig-id.nb
    c2i2 = np.cos(0.5*inc)**2
    s2i2 = np.sin(0.5*inc)**2
    arg0 = truan + lan
    arg1 = truan + argp + lan
    arg2 = truan + argp - lan
    arg3 = truan - lan
    c1 = np.cos(arg1)
    c2 = np.cos(arg2)
    s1 = np.sin(arg1)
    s2 = np.sin(arg2)
    sa0 = np.sin(arg0)
    sa3 = np.sin(arg3)
    # updated sign convention for Green Eq. 19.4-19.7
    xp = radius*(c2i2*s1 - s2i2*s2)
    yp = radius*(c2i2*c1 + s2i2*c2)
    # Interleave x & y
    # put x data in odd elements and y data in even elements
    data = np.zeros(ndate)
    data[0::2]  = xp
    data[1::2]  = yp
    return data*parallax # results in seconds of arc

def logl(theta,t,x,y,sig_x,sig_y):
    loga = theta[0]
    tau = theta[1]
    argp = theta[2]
    lan = theta[3]
    cinc = theta[4]
    ecc = theta[5]
    
    a = np.exp(loga)
    inc = np.arccos(cinc)
    
    X,Y = koe(t,a,tau,argp,lan,inc,ecc)
    
    lpx = -0.5*np.log(2*np.pi)*x.size*2. - np.sum(np.log(sig_x)+np.log(sig_y)+ \
                    0.5*(((x-X)/sig_x)**2+((y-Y)/sig_y)**2))
                    
    return lpx
    
def logp(theta):
    loga = theta[0]
    tau = theta[1]
    argp = theta[2]
    lan = theta[3]
    cinc = theta[4]
    ecc = theta[5]

    # include priors here

    if (loga < np.log(80) or loga > np.log(800)): 
        return -np.inf

    if (tau < -0.5 or tau > 0.5):
        return -np.inf
 
    if (argp < 0 or argp > 2*np.pi):
        return -np.inf

    if (lan < np.pi/18.*11. or lan > np.pi/9.*10.):
        return -np.inf

    if (cinc < 0 or cinc > 1): 
        return -np.inf

    #if (ecc < 0 or ecc >= 0.95724):
    if (ecc <0.4 or ecc >= 1.0):
        return -np.inf
   
    # otherwise ... 
    return 0.
    
ntemps   = 20 #Number of temperatures for the parallel tempering
nwalkers = 128 #Number of walkers
ndim     = 6 #Number of dimensions in the orbit fit

#filename='51eri.csv' #Beta Pic
filename='fomalhaut_new.csv' #Fomalhaut
eps=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,10), unpack=True)).T.ravel()
xy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(11,13), unpack=True)).T.ravel()/1000
sigs=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(12,14), unpack=True)).T.ravel()/1000
eps=eps[xy == xy]
sigs=sigs[xy == xy]
xy=xy[xy == xy]

#'''
#FOR FOMALHAUT
w0 = np.random.uniform( 4.0,7.0,    size=(ntemps,nwalkers))
w1 = np.random.uniform( -0.25,0.25,    size=(ntemps,nwalkers))
w2 = np.random.uniform(-np.pi/4.,3.*np.pi/4.,  size=(ntemps,nwalkers))
w3 = np.random.uniform( np.pi/3,np.pi,    size=(ntemps,nwalkers))
w4 = np.random.uniform(-0.3,0.3,  size=(ntemps,nwalkers))
w5 = np.random.uniform( 0.785,0.825,   size=(ntemps,nwalkers))

p0 = np.dstack((w0,w1,w2,w3,w4,w5))

#Numnber of iterations (after the burn-in)
niter = 250000
num_files=10 #I trust YOU, user, to make sure niter / num_files is remainderless

nperfile=niter/num_files

#Burn-in for 1/5 of the number of iterations
nburn = 50000

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

    
    print 'Total Execution Time:', (datetime.now()-startTime)