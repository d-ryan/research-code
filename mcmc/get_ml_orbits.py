import numpy as np
import csv
from emcee import autocorr
from datetime import date
from jdcal import gcal2jd

planets = ['b','c','d','e']

with open('hr8799_params.csv','wb') as csvfile:
    writer = csv.writer(csvfile)
    row = ['HR8799','a (AU)','e','i (rad)','argp (rad)','lan (rad)','tau(MJD)','m_pl(Mj)','mstar(Ms)','dist(pc)']
    writer.writerow(row)

    for p in planets:

        chains = np.zeros((128.,250000.,6.))
        acs = np.zeros((6.,10.))
        lnps = np.zeros((128.,250000.))
        for i in range(10):
            with np.load('F:/mcmc/hr8799/'+p+'/test_'+str(i)+'.npz') as d:
                chain = d['chain']
                ac = d['ac']
                lnp = d['lnp']
            chains[:,25000*i:25000*(i+1),:]=chain
            acs[:,i]=ac
            lnps[:,25000*i:25000*(i+1)]=lnp
        thinf = np.ceil(np.nanmax(acs))
        lnaflat = chains[:,:,0].flatten()[::thinf]
        lnpflat = lnps.flatten()[::thinf]
        tauflat = chains[:,:,1].flatten()[::thinf]
        argpflat = chains[:,:,2].flatten()[::thinf]
        lanflat = chains[:,:,3].flatten()[::thinf]
        cincflat = chains[:,:,4].flatten()[::thinf]
        eflat = chains[:,:,5].flatten()[::thinf]
        
        w = np.argmax(lnpflat)
        
        lna_ml=lnaflat[w]
        tau_ml=tauflat[w]
        argp_ml=argpflat[w]
        lan_ml=lanflat[w]
        cinc_ml=cincflat[w]
        ecc_ml=eflat[w]
        
        if p=='b': m_pl = 4.5
        else: m_pl = 7.
        
        per = np.sqrt(np.exp(lna_ml)**3./1.51)*365.25636
        
        todaydt = date.today()
        
        todaymjd = gcal2jd(todaydt.year, todaydt.month, todaydt.day)[1]
        
        #tau from chains is in units of orbital period.
        #Convert to MJD by multiplying by period, adding 50000
        #Limit range of dates by taking modulo period
        #Subtract today's MJD inside modulo and add outside: yields dates from today to today+per
        #subtract per outside modulo: yields dates from today-per to today: MOST RECENT epoch of peri
        tau_ml = (tau_ml*per+50000-todaymjd)%per + todaymjd - per
        
        row = [p,np.exp(lna_ml),ecc_ml,np.arccos(cinc_ml),argp_ml,lan_ml,tau_ml,m_pl,1.51,36.4]
        writer.writerow(row)