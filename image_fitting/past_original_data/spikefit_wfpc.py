import numpy as np
from astropy.io import fits
from skimage.transform import radon
from skimage.transform._warps_cy import _warp_fast
from scipy.ndimage import map_coordinates,gaussian_filter
from scipy.optimize import curve_fit
import pdb

'''CRITICAL NOTE'''
#Filtered images: vertical: all
#horizontal: all
'''END CRITICAL NOTE'''


#the following command sequence prevents lag when drawing plots on windows.
#must be done in this order: need matplotlib but none of its sub-modules
#or sub-packages imported in order to call matplotlib.use()
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

location='F:/BetaPic/wfpc2/'
prefixlist=['hst_05202','hst_05700','hst_06058','hst_06253']
numlist=[['01','02','03','04'],['01','02','03','04'],['01','02','03','04'],['01','02','03','04']]
suffix='_wfpc2_f'
wvlist=['160b','255','439','555','675','814']
suffix2='w_pc_drz.fits'

pairs=[(0,1,2,3,4,5),(2,3,4,5),(1,2),(0,),(2,),(1,),(0,2),(1,2)]

#Right: 2822 @3000, 2821 @4640
#Left: 2821 @2720, 2821


#909,942 - 1092,1306
#2420,3244 - 2436,3173

def choosefit(img,fitfn):
    shp=img.shape[0]
    inds=np.arange(shp)
    cen=shp//2
    c=cen-40
    
    badsx=[]
    badsy=[]
    
    paramsx=np.zeros((3.,shp))
    covsx=np.zeros((3.,3.,shp))
    paramsy=np.zeros((3.,shp))
    covsy=np.zeros((3.,3.,shp))
    altx=np.zeros(img.shape)
    alty=np.zeros(img.shape)
    
    for i in inds:
        nz=np.where(img[:,i]>0)[0]
        imod=i-cen
        offset=img[nz,i]-np.mean(img[nz,i])
        
        p0=(6,520.,.05)
        big=nz[np.where(offset>10)[0]]
        if len(big)>0:
            fitcoords=np.arange(int(p0[1])-40,int(p0[1])+40,1)
            
            fitvals=img[fitcoords,i]-np.mean(img[nz,i])
            
            try:
                #p0=(fitvals.max(),520.,.05)
                p0=(fitvals.max(),515-10.*imod/float(shp),.05)
                paramsx[:,i],covsx[:,:,i]=curve_fit(fitfn,fitcoords,fitvals,p0=p0,sigma=np.sqrt(np.abs(fitvals))/np.abs(fitvals))
                altx[:,i]=fitfn(inds,*paramsx[:,i])
            except:
                badsx.append(i)
            else:
                if paramsx[2,i]>10:
                    badsx.append(i)
        else:
            try:
                paramsx[:,i],covsx[:,:,i]=curve_fit(fitfn,nz[2:-2],offset[2:-2],p0=p0)
                altx[:,i]=fitfn(inds,*paramsx[:,i])
            except:
                badsx.append(i)
            else:
                if paramsx[2,i]>10:
                    badsx.append(i)

        print i
    return paramsx,covsx,altx,badsx

def rotator(theta,center):
    shift0=np.array([[1,0,-center],[0,1,-center],[0,0,1]])
    shift1=np.array([[1,0,center],[0,1,center],[0,0,1]])
    R=np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
    return shift1.dot(R).dot(shift0)

def get_data(filename):
    hdulist=fits.open(filename)
    rawdata=hdulist[1].data
    hdulist.close()
    return rawdata
    
    
def sinc2(x,a,p,s):
    f=a*np.sinc(s*(x-p))
    return f**2
    
    
def routine(filename,savename):
    data = get_data(filename)

    px,cx,ax,bx=choosefit(data,sinc2)
    
    #Save these parameters in case they're not worthless
    np.savez_compressed(savename,p=px,c=cx,b=bx,a=ax)
    return
    
def check(fname,sname):
    with np.load(sname+'.npz') as d:
        p=d['p']
    pos=get_data(fname)
    poscopy=pos.copy()
    poscopy[poscopy>1200]=1200
    poscopy[poscopy<2]=2
    plt.imshow(np.log(poscopy),origin='lower')
    inds=np.arange(0.,float(pos.shape[0]),1.)
    plt.plot(inds,p[1,:],'ro')
    plt.show()

    
if __name__=='__main__':
    for k,prefix in enumerate(prefixlist):
        for i,n in enumerate(numlist[k]): #Image number
            for j in pairs[k][i]: #Wavelength
                #if j<2: continue
                fname = location+prefix+'_'+n+suffix+wvlist[j]+suffix2
                sname = location+'/'+prefix+'/'+n+'/horiz'+'_'+wvlist[j]

                routine(fname,sname)
                check(fname,sname)