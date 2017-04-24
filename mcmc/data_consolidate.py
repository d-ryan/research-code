import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from fmh_constants import constants
import csv


parallax,distance,mstar,kgauss, AU, DAY = constants()


pickle_filename = 'F:/mcmc/fmh_new_longtest1'

num_files=10
nperfile=25000
acceptance_fraction = np.zeros((128*num_files))
acor = np.zeros((6*num_files))
chain = np.zeros((128,nperfile*num_files,6))
lnprobability = np.zeros((128,nperfile*num_files))

for i in range(num_files):
    with np.load(pickle_filename+'_'+str(i)+'.npz') as d:
        acceptance_fraction[128*i:128*(i+1)] = d['af']
        acor[6*i:6*(i+1)] = d['ac']
        chain[:,nperfile*i:nperfile*(i+1),:]=d['chain']
        lnprobability[:,nperfile*i:nperfile*(i+1)]=d['lnp']
        
#np.savez_compressed(pickle_filename+'_master',af=acceptance_fraction,ac=acor,chain=chain,lnp=lnprobability)


P = np.sqrt(np.exp(chain[:,:,0])**3./mstar)*365.2425

chain[:,:,1]=((chain[:,:,1]*P)+50000-49718)%P+49718

SMA = np.exp(np.percentile(np.ndarray.flatten(chain[:,:,0]),[5,16,50,84,95]))
TAU = np.percentile(np.ndarray.flatten(chain[:,:,1]),[5,16,50,84,95])
ARGP= np.degrees(np.percentile(np.ndarray.flatten(chain[:,:,2]),[5,16,50,84,95]))
LAN = np.degrees(np.percentile(np.ndarray.flatten(chain[:,:,3]),[5,16,50,84,95]))
INC = np.degrees(np.arccos(np.percentile(np.ndarray.flatten(chain[:,:,4]),[5,16,50,84,95])))
ECC = np.percentile(np.ndarray.flatten(chain[:,:,5]),[5,16,50,84,95])
PER = 2*np.pi/(kgauss*np.sqrt(mstar)*(SMA)**(-1.5) ) /365.25

percentile_arr=np.zeros((5,7))
percentile_arr[:,0]=SMA
percentile_arr[:,1]=TAU
percentile_arr[:,2]=ARGP
percentile_arr[:,3]=LAN
percentile_arr[:,4]=INC
percentile_arr[:,5]=ECC
percentile_arr[:,6]=PER

thinf = np.ceil(np.nanmax(acor))

lna  = chain[:,:,0].flatten()
tau  = chain[:,:,1].flatten()
argp = chain[:,:,2].flatten()
lan  = chain[:,:,3].flatten()
cinc = chain[:,:,4].flatten()
ecc  = chain[:,:,5].flatten()

lnp =  lnprobability.flatten()
w = np.argmax(lnp)

peri = np.exp(lna)*(1.-ecc)
apo = np.exp(lna)*(1.+ecc)

#Get the maximum likely
a_ml=np.exp(lna[w])
tau_ml=tau[w]
argp_ml=np.degrees(argp[w])
lan_ml=np.degrees(lan[w])
inc_ml=np.degrees(np.arccos(cinc[w]))
ecc_ml=ecc[w]
per_ml=2*np.pi/(kgauss*np.sqrt(mstar)*a_ml**(-1.5))/365.2425

ml_vec = np.array([a_ml,tau_ml,argp_ml,lan_ml,inc_ml,ecc_ml,per_ml])

a_mean = np.mean(np.exp(lna))
tau_mean = np.mean(tau)
argp_mean = np.mean(np.degrees(argp))
lan_mean = np.mean(np.degrees(lan))
inc_mean = np.mean(np.degrees(np.arccos(cinc)))
ecc_mean = np.mean(ecc)

mean_vec = np.array([a_mean,tau_mean,argp_mean,lan_mean,inc_mean,ecc_mean])

a_std = np.std(np.exp(lna))
tau_std = np.std(tau)
argp_std = np.std(np.degrees(argp))
lan_std = np.std(np.degrees(lan))
inc_std = np.std(np.degrees(np.arccos(cinc)))
ecc_std = np.std(ecc)
std_vec = np.array([a_std,tau_std,argp_std,lan_std,inc_std,ecc_std])

a_rms = np.sqrt(np.mean(np.exp(lna)**2.))
tau_rms = np.sqrt(np.mean(tau**2.))
argp_rms = np.sqrt(np.mean(np.degrees(argp)**2.))
lan_rms = np.sqrt(np.mean(np.degrees(lan)**2.))
inc_rms = np.sqrt(np.mean(np.degrees(np.arccos(cinc))**2.))
ecc_rms = np.sqrt(np.mean(ecc**2.))
rms_vec = np.array([a_rms,tau_rms,argp_rms,lan_rms,inc_rms,ecc_rms])

print "Percentiles"
print SMA,'AU'
print TAU,'MJD'
print ARGP,'deg'
print LAN,'deg'
print INC,'deg'
print ECC
print PER,'Yr'
print " "
print "Max Likelihood"
print a_ml,'AU'
print tau_ml,'MJD'
print argp_ml,'deg'
print lan_ml,'deg'
print inc_ml,'deg'
print ecc_ml
print per_ml,'Yr'

with open('fmh_new_priors.csv','wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['value','a','tau','argp','lan','inc','ecc'])
    writer.writerow(['5%']+list(percentile_arr[0,:-1]))
    writer.writerow(['16%']+list(percentile_arr[1,:-1]))
    writer.writerow(['50%']+list(percentile_arr[2,:-1]))
    writer.writerow(['84%']+list(percentile_arr[3,:-1]))
    writer.writerow(['95%']+list(percentile_arr[4,:-1]))
    writer.writerow(['mean']+list(mean_vec))
    writer.writerow(['std']+list(std_vec))
    writer.writerow(['rms']+list(rms_vec))
    
np.savez_compressed(pickle_filename+'_likely',percentile_arr=percentile_arr,ml_vec=ml_vec,mean_vec=mean_vec,std_vec=std_vec,rms_vec=rms_vec)