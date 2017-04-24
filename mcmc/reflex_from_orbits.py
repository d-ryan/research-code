import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
import triangle
import pdb
from koe_m import koe

# from constants import constants

import jdcal as jdcal
import datetime as dt
import random
from datetime import date

from orbitclass import orbit

from reflex_m import make_times,time_plots



filename='hd206893.csv'
epochs=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,10), unpack=True)).T.ravel()
xy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(11,13), unpack=True)).T.ravel()/1000
sig=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(12,14), unpack=True)).T.ravel()/1000

pickle_filename = 'F:/mcmc/hd206893/test'
#pickle_filename = 'fmh_new_priorstest2'

num_files=5
nperfile=25000
acceptance_fraction = np.zeros((128*num_files))
acor = np.zeros((8*num_files))
chain = np.zeros((128,nperfile*num_files,8))
lnprobability = np.zeros((128,nperfile*num_files))

for i in range(num_files):
    with np.load(pickle_filename+'_'+str(i)+'.npz') as d:
        acceptance_fraction[128*i:128*(i+1)] = d['af']
        acor[8*i:8*(i+1)] = d['ac']
        chain[:,nperfile*i:nperfile*(i+1),:]=d['chain']
        lnprobability[:,nperfile*i:nperfile*(i+1)]=d['lnp']
        
tmp=chain[:,:,1]
tmp[tmp > 0.5] -= 1.
chain[:,:,1]=tmp

#We'll wrap argp about 360 degrees
tmp=chain[:,:,2]
tmp[tmp > np.radians(360)] -= np.radians(360)
chain[:,:,2]=tmp

#Thin it out by the maximum autocorrelation
thinf = np.ceil(np.nanmax(acor))

lna  = chain[:,:,0].flatten()[::thinf]
tau  = chain[:,:,1].flatten()[::thinf]
argp = chain[:,:,2].flatten()[::thinf]
lan  = chain[:,:,3].flatten()[::thinf]
cinc = chain[:,:,4].flatten()[::thinf]
ecc  = chain[:,:,5].flatten()[::thinf]
mstar= chain[:,:,6].flatten()[::thinf]
dist = chain[:,:,7].flatten()[::thinf]

#Where is the maximum probability? 
lnp =  lnprobability.flatten()[::thinf]
w = np.argmax(lnp)

P = np.sqrt(np.exp(lna)**3./mstar)*365.256
tau = ((tau*P)+50000-49718)%P+49718

#Get the maximum likely
lna_ml=lna[w]
tau_ml=tau[w]
argp_ml=argp[w]
lan_ml=lan[w]
cinc_ml=cinc[w]
ecc_ml=ecc[w]
mstar_ml=mstar[w]
dist_ml=dist[w]

#Calculate the inclination in degrees
inc=np.arccos(cinc)
inc_ml=np.arccos(cinc_ml)

mpl_ml = 50.

cands = np.arange(len(lna))
#a e i argp lan tau m_pl m_star dist
orbitlist = []

#inds_lnp = np.argpartition(lnp,-)

for k in range(100):
    i = random.choice(cands)
    a = np.exp(lna[i])
    e = ecc[i]
    inclin = inc[i]
    w = argp[i]
    O = lan[i]
    epperi = tau[i]
    mpl = random.gauss(49.,2.5)
    if mpl<1: mpl = 1.
    m_star = mstar[i]
    d = dist[i]
    orbitlist.append(orbit(a=a,e=e,i=inclin,argp=w,lan=O,tau=epperi,m_planet=mpl,m=m_star,dist=d))

orbitlist.append(orbit(a=np.exp(lna_ml),e=ecc_ml,i=inc_ml,argp=argp_ml,lan=lan_ml,tau=tau_ml,m_planet=mpl_ml,m=mstar_ml,dist=dist_ml))

n_years = 30
start = jdcal.gcal2jd(date.today().year,date.today().month,date.today().day)[1]
times=make_times(start,start-365.2425*n_years,(n_years*24.)+1.)

time_plots(times,orbitlist)