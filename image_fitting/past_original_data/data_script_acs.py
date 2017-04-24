import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Have consolidated data for 01 thru 09.1. Need to do for 09.2-10.
#Need to try out the outlier-resistant linear fit.


fnames = ['j70s','j71s','j72s','j73s']

rawdata=np.zeros((4,2,2462)) #4 images, 2*2 spikes, 2 params each, 2462 pixels
trimdata=np.zeros((2,2,2462)) #4 images, 2 spikes, 2 params each, 2462 pixels

def line(x,m,b):
    return m*x+b

def parab(x,a,b,c):
    return a*x**2 + b*x + c

def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

linfits=np.zeros((4,8,2))
parfits=np.zeros((4,8,3))
cubfits=np.zeros((4,8,4))

#Image ordering: j70s, j71s, j72s, j73s
#Spike ordering: sorted first by side, then by number w/in side
#                left, right, top, bottom
#                spike 1 then spike 2
#Parameter ordering: max, sigma
#

location='F:/BetaPic/acs/raw/'
prefix='hst_9861_'
numlist=['01','04','05','06','07','08','09','10']
suffix='_acs_wfc_f'
wvlist=['435','606','814']
suffix2='w_drz.fits'

pairs=[(0,1),(0,2),(1,2),(0,),(2,),(1,),(0,2),(1,2)]

side=['LEFT','RIGHT','TOP','BOT']



if __name__=='__main__':
    for i,n in enumerate(numlist): #Image number
        for j in pairs[i]: #Wavelength
            if i<7: continue
            if j==0: continue
            sname = location+n+'/spike_master_'+n+'_'+wvlist[j]
            for k in range(4): #Spike: Left, right, top, bottom
                fname = location+n+'/'+side[k]+n+'_'+wvlist[j]+'.npz'
                with np.load(fname) as d:
                    px=d['p']
                max1=px[1,:]
                sig1=px[2,:]
                img_size=np.size(max1)
                if k==0:
                    rawdata=np.zeros((4,2,img_size)) #2*2 spikes, 2 params each, N pixels
                    trimdata=np.zeros((2,2,img_size)) #2 spikes, 2 params each, N pixels
            
                inds=np.arange(0.,float(img_size),1.)
                intinds=np.arange(0,int(img_size),1)
                
                rawdata[k,0,:] = max1.copy()
                #rawdata[2*i+1,0,:] = max2.copy()
                rawdata[k,1,:] = sig1.copy()
                #rawdata[2*i+1,1,:] = sig2.copy()
                
                #Check and make sure gaussian maxima are fine
                #Manually throw out weird data
                #While keeping a record of where the data goes
                while True:
                    plt.plot(inds,max1)
                    plt.show()
                    rem=raw_input('Remove data? ')
                    if rem=='': break
                    elif rem[0]=='y' or rem[0]=='Y':
                        front=int(raw_input('Front index: '))
                        back=int(raw_input('Back index: '))
                        ind1=np.where(inds==front)[0][0]
                        if back==-1: ind2=np.max(inds)
                        else: ind2=np.where(inds==back)[0][0]
                        mask=np.ones(len(inds))
                        mask[ind1:ind2]=0
                        inds=inds[mask==1]
                        intinds=intinds[mask==1]
                        max1=max1[mask==1]
                        #max2=max2[mask==1]
                        sig1=sig1[mask==1]
                        #sig2=sig2[mask==1]
                    else:
                        try:
                            intrem=int(rem)
                        except ValueError:
                            break
                        else:
                            continue
            
                while True:
                    plt.plot(inds,sig1)
                    plt.show()
                    rem=raw_input('Remove data? ')
                    if rem=='': break
                    elif rem[0]=='y' or rem[0]=='Y':
                        front=int(raw_input('Front index: '))
                        back=int(raw_input('Back index: '))
                        ind1=np.where(inds==front)[0][0]
                        if back==-1: ind2=np.max(inds)
                        else: ind2=np.where(inds==back)[0][0]
                        mask=np.ones(len(inds))
                        mask[ind1:ind2]=0
                        inds=inds[mask==1]
                        intinds=intinds[mask==1]
                        max1=max1[mask==1]
                        sig1=sig1[mask==1]
                    else:
                        try:
                            intrem=int(rem)
                        except ValueError:
                            break
                        else:
                            continue
                            
                            
                if k<2: zz=0
                else: zz=1
                
                trimdata[zz,0,intinds] = max1.copy()
                trimdata[zz,1,intinds] = sig1.copy()
                
            np.savez_compressed(sname,raw=rawdata,trim=trimdata)
            print 'saved',sname
            
            

'''
for j,f in enumerate(fnames): #operate on all 4 images
    full = f+'.npz'
    files = ['posLEFT'+full,'posRT'+full,'posTOP'+full,'posBOT'+full]
    for i,file in enumerate(files): #And on all four sides
                with np.load(file) as d:
                    px=d['p']
                max1=px[1,:]
                sig1=px[2,:]
                inds=np.arange(0.,float(np.size(max1)),1.)
                intinds=np.arange(0,int(np.size(max1)),1)
                
                rawdata[j,2*i,0,:] = max1.copy()
                rawdata[j,2*i+1,0,:] = max2.copy()
                rawdata[j,2*i,1,:] = sig1.copy()
                rawdata[j,2*i+1,1,:] = sig2.copy()
                
                #Check and make sure gaussian maxima are fine
                #Manually throw out weird data
                #While keeping a record of where the data goes
                while True:
                    plt.plot(inds,max1,inds,max2)
                    plt.show()
                    rem=raw_input('Remove data? ')
                    if rem=='': break
                    elif rem[0]=='y' or rem[0]=='Y':
                        front=int(raw_input('Front index: '))
                        back=int(raw_input('Back index: '))
                        ind1=np.where(inds==front)[0][0]
                        if back==-1: ind2=np.max(inds)
                        else: ind2=np.where(inds==back)[0][0]
                        mask=np.ones(len(inds))
                        mask[ind1:ind2]=0
                        inds=inds[mask==1]
                        intinds=intinds[mask==1]
                        max1=max1[mask==1]
                        max2=max2[mask==1]
                        sig1=sig1[mask==1]
                        sig2=sig2[mask==1]
                    else:
                        try:
                            intrem=int(rem)
                        except ValueError:
                            break
                        else:
                            continue
                
                #Do a similar check on std dev; throw out weirds manually
                while True:
                    plt.plot(inds,sig1,inds,sig2)
                    plt.show()
                    rem=raw_input('Remove data? ')
                    if rem=='': break
                    elif rem[0]=='y' or rem[0]=='Y':
                        front=int(raw_input('Front index: '))
                        back=int(raw_input('Back index: '))
                        ind1=np.where(inds==front)[0][0]
                        if back==-1: ind2=np.max(inds)
                        else: ind2=np.where(inds==back)[0][0]
                        mask=np.ones(len(inds))
                        mask[ind1:ind2]=0
                        inds=inds[mask==1]
                        intinds=intinds[mask==1]
                        max1=max1[mask==1]
                        max2=max2[mask==1]
                        sig1=sig1[mask==1]
                        sig2=sig2[mask==1]
                    else:
                        try:
                            intrem=int(rem)
                        except ValueError:
                            break
                        else:
                            continue
                
                pl1,cl1=curve_fit(line,inds,max1)
                pp1,cp1=curve_fit(parab,inds,max1)
                pc1,cc1=curve_fit(cubic,inds,max1)

                pl2,cl2=curve_fit(line,inds,max2)
                pp2,cp2=curve_fit(parab,inds,max2)
                pc2,cc2=curve_fit(cubic,inds,max2)
                
                #linfits=np.zeros((4,8,2))
                #parfits=np.zeros((4,8,3))
                #cubfits=np.zeros((4,8,4))
                
                linfits[j,2*i,:]=pl1
                parfits[j,2*i,:]=pp1
                cubfits[j,2*i,:]=pc1

                linfits[j,2*i+1,:]=pl2
                parfits[j,2*i+1,:]=pp2
                cubfits[j,2*i+1,:]=pc2
                
                if i<2: k=0
                else: k=2
                
                trimdata[j,k,0,intinds] = max1.copy()
                trimdata[j,k+1,0,intinds] = max2.copy()
                trimdata[j,k,1,intinds] = sig1.copy()
                trimdata[j,k+1,1,intinds] = sig2.copy()
                
                np.savez_compressed('spikedata',raw=rawdata,trim=trimdata,lin=linfits,par=parfits,cub=cubfits)
'''