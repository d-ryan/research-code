
#############################################
import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pylab as P
from matplotlib.ticker import MaxNLocator
import triangle
import pdb
from koe_m import koe

# from constants import constants

import jdcal as jdcal
import datetime as dt
#############################################


#Read in the astrometry datapoints
#filename='astrometry_csv.csv'
filename='fomalhaut_new.csv'
#filename='51eri.csv'
filename='hr8799'
filename='hd206893.csv'
epochs=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,10), unpack=True)).T.ravel()
xy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(11,13), unpack=True)).T.ravel()/1000
sig=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(12,14), unpack=True)).T.ravel()/1000

#Read in the pickle:
#pickle_filename= 'pt_ntmp_20_nwlkrs_128_lgniter_3.78_V_310.319485.pkl' #This one has no gpi data (max's good beta pic)
#pickle_filename= 'pt_ntmp_20_nwlkrs_128_lgniter_4.00_V_1626.421915.pkl' #(GOod?) Fomalhaut
#pickle_filename= 'pt_ntmp_20_nwlkrs_128_lgniter_3.00_V_513.505076.pkl' # (bad bpic b/c burn in not removed)
#pickle_filename= 'pt_ntmp_20_nwlkrs_128_lgniter_3.00_V_303.247270.pkl' #(my good bpic, agrees w/ max's)

#pickle_filename = 'es_nwlkrs_128_lgniter_4.30_V_224.291257.pkl' #ensemble fomalhaut -bad
#pickle_filename = 'es_nwlkrs_128_lgniter_5.60_V_3971.984152.pkl' #LONG ensemble fomalhaut - bad
#pickle_filename = 'pt_ntmp_20_nwlkrs_128_lgniter_5.00_V_27386.096664.pkl' #VERY LONG PT fomalhaut -good??
#pickle_filename = 'pt_ntmp_20_nwlkrs_128_lgniter_5.00_V_15040.692732.pkl' #PT fomalhaut w/ new data
#pickle_filename = 'pt_ntmp_20_nwlkrs_128_lgniter_5.30_V_33113.517884.pkl' #200k PT fomalhaut, 4 epochs
#pickle_filename = 'pt_ntmp_20_nwlkrs_128_lgniter_4.78_V_8591.735577.pkl' #51 eri, possibly crap
#pickle_filename = 'pt_ntmp_20_nwlkrs_128_lgniter_4.30_V_1718.063888.pkl' #Okay 51 eri

#10/6 tests
#pickle_filename = 'pt_ntmp_20_nwlkrs_128_lgniter_3.08_V_102.756010.pkl' 
#pickle_filename = 'pt_ntmp_20_nwlkrs_128_lgniter_3.08_V_59.406167.pkl'

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
'''    
with np.load(pickle_filename) as d:
    acceptance_fraction = d['af']
    acor = d['ac']
    chain = d['chain']
    lnprobability = d['lnp']
'''
print np.shape(acceptance_fraction)
print np.shape(acor)
print np.shape(chain)
print np.shape(lnprobability)
#pdb.set_trace()


#Let's wrap around periodic values: 

#We'll wrap tau about 1 period
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
inc=np.degrees(np.arccos(cinc))
inc_ml=np.degrees(np.arccos(cinc_ml))

print 'Raw max like values----------'
print 'SMA ',lna_ml
print 'TAU ', tau_ml
print 'ARGP', argp_ml
print 'LAN ', lan_ml
print 'CINC',cinc_ml
print 'ECC ',ecc_ml
print 'MSTAR',mstar_ml
print 'DIST',dist_ml
print '-------------------------'


print 'Max like values----------'
print 'SMA ',np.exp(lna_ml)
print 'LAN ',np.degrees(lan_ml)
print 'ECC ', ecc_ml
print 'INC' , inc_ml
print '-------------------------'


#Reformat the chain to useful units
chain[:,:,0]=np.exp(chain[:,:,0]) #Get the SMA in AU
chain[:,:,2]=np.degrees(chain[:,:,2]) # In degrees
chain[:,:,3]=np.degrees(chain[:,:,3]) # In degrees
chain[:,:,4]=np.degrees(np.arccos(chain[:,:,4])) #in degrees

P = np.sqrt(chain[:,:,0]**3./chain[:,:,6])*365.2425

chain[:,:,1]=((chain[:,:,1]*P)+50000-49718)%P+49718

#######################################################
#################### Make the plot ####################
#######################################################
ndim=8

# samples=chain[:,:,:].reshape(-1,6)
samples=chain[:,:,:].reshape(-1,ndim)

labels=["a[AU]",r'$\tau$',r'$\omega[^{\circ}]$',r'$\Omega[^{\circ}]$',r'$i[^{\circ}]$',r'$e$',r'$m$',r'$d$']

fig=triangle.corner(samples, labels=labels, show_titles=True,plot_datapoints=False)

fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)

####################################################################
#################### Make the inset plot like Chauvin ##############
####################################################################

#Stick the plot in the top right
plt2=fig.add_subplot(3,3,3)

#### Generate Epochs for the lines ####

#Set up
DT = 50000.        # my time offset from MJD
MJD0 = 2400000.5   # zero point for JD
# generate range of dates 
epo_pad = 365*7
epo = np.linspace(np.nanmin(epochs)-epo_pad,np.nanmax(epochs)+epo_pad,num=100)

epo = np.column_stack((epo,epo)).flatten()
#convert epochs to datetime stucture
ddt = np.array([])
for ep in epo:
    y,m,d,h = jdcal.jd2gcal(MJD0 + DT, ep)
    ddt = np.append(ddt,dt.datetime(y,m,d))
# convert observed dates for plotting
epochdt = np.array([])

##### Convert the Data epochs #### 

for ep in epochs:
    y,m,d,h = jdcal.jd2gcal(MJD0 + DT, ep)
    epochdt = np.append(epochdt,dt.datetime(y,m,d))


#Plot a horizontal line at 0
plt.axhline(y=0,ls='--',color='k')

#Get the orbit and plot it for npts
npts = 50
rindex = np.floor(np.random.random(npts) * np.size(lna))
for i in rindex:
        XY = koe(epo,np.exp(lna[i]),tau[i],argp[i],lan[i],np.arccos(cinc[i]),ecc[i],mstar[i],dist[i])
        plt2.plot(ddt[0::2],XY[0::2],'r:',linewidth=1, alpha=0.05)
        plt2.plot(ddt[1::2],XY[1::2],'b:',linewidth=1, alpha=0.05)


#Plot the most likely solution
XY=koe(epo,np.exp(lna_ml),tau_ml,argp_ml,lan_ml,np.arccos(cinc_ml),ecc_ml,mstar_ml,dist_ml)

plt.plot(ddt[0::2],XY[0::2],'k',linewidth=1.5)
plt.plot(ddt[1::2],XY[1::2],'k',linewidth=1.5)

#### Plot the position of the planet ####

plt2.errorbar(epochdt[0::2],xy[0::2],yerr=sig[0::2],fmt='ro',ecolor='k', elinewidth=2, capthick=2, ms=2)
plt2.plot(epochdt[0::2],xy[0::2],'ro',mew=1.5,label=r'X')
plt2.legend(numpoints=1,markerscale=2)

plt2.errorbar(epochdt[1::2],xy[1::2],yerr=sig[1::2],fmt='bo',ecolor='k', elinewidth=2, capthick=2, ms=2)
plt2.plot(epochdt[1::2],xy[1::2],'bo',mew=1.5,label=r'Y')
plt2.legend(numpoints=1,markerscale=2)


### Set titles and limits ###
plt.xlim(dt.datetime(2008, 1,1),dt.datetime(2025, 1,1))
plt.ylabel('Offset [arcsec]')
plt.xlabel('Date [year]')

#Get the output filename for the figure
tmp=pickle_filename.split('/')
outname=tmp[-1]

#Save the figure
plt.savefig(outname+'.png')

plt.show()