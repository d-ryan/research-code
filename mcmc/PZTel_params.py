import numpy as np


#Filename for astrometry
filename = 'F:/mcmc/PZTel/data.csv'

#Location to save data
savename = 'F:/mcmc/PZTel/hyp/test'

#Temperatures of walkers to save
savetemps = [0,4,9,14,19]

#MCMC bookkeeping
ntemps   = 20
nwalkers = 128
ndim     = 8

'''
nper     = 2500
n_burns  = 1
n_files  = 1
'''
nper = 2500
n_burns = 1
n_files = 1

#Min & max bounds for flat priors (on loga, tau, argp, )
a_bounds = [1,500]
tau_bounds = [-0.5,0.5]
argp_bounds = [0.,2.*np.pi]
lan_bounds = [0.,2.*np.pi]
cinc_bounds = [-1.,1.]
ecc_bounds = [0.,1.]
mass_params = [1.13,0.03]
dist_params = [51.49,2.60]

prior_bounds = [a_bounds,tau_bounds,argp_bounds,lan_bounds,cinc_bounds,ecc_bounds,mass_params,dist_params]


parameters = [ntemps,nwalkers,ndim,nper,n_burns,n_files,filename,savename,savetemps,prior_bounds]
