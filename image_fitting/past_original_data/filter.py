import numpy as np
from scipy.fftpack import fft2,ifft2
from astropy.io import fits

def highpass(data,maskargs):
    mask = np.zeros(data.shape)
    xshp = data.shape[0]
    xcen = xshp//2
    yshp = data.shape[1]
    ycen = yshp//2
    fft_data = fft2(data)
    for i in range(xshp):
        for j in range(yshp):
            mask[i,j] = hpmask(xcen,ycen,i,j,*maskargs)
    return mask*data
            
            
def lowpass(data):
    mask = np.zeros(data.shape)
    xshp = data.shape[0]
    xcen = xshp//2
    yshp = data.shape[1]
    ycen = yshp//2
    fft_data = fft2(data)
    for i in range(xshp):
        for j in range(yshp):
            mask[i,j] = lpmask(xcen,ycen,i,j,xshp/4.,0.1*xshp/4.)
    return mask*data
            
            
            
def lpmask(x,y,i,j,r,w):
    xoff = x-i
    yoff = y-j
    d = np.sqrt(xoff**2 + yoff**2)
    if d<r-w: return 1
    elif d>r+w: return 0
    else: return 0.5*(1-np.sin(np.pi*(d-r)/2./w))
    
def hpmask(x,y,i,j,r,w):
    xoff = x-i
    yoff = y-j
    d = np.sqrt(xoff**2 + yoff**2)
    if d<r-w: return 0
    elif d>r+w: return 1
    else: return 0.5*(np.sin(np.pi*(d-r)/2./w)-1.)

def get_data(filename):
    hdulist=fits.open(filename)
    rawdata=hdulist[1].data
    hdulist.close()
    return rawdata
    
def prep_filt(filename,savename):
    dat=get_data(filename)
    dat=dat.astype('float')
    transformed = fft2(dat)
    xshp = dat.shape[0]
    maskargs = [xshp*0.4,xshp*0.12]
    hp = highpass(transformed,maskargs)
    recon = ifft2(hp)
    np.savez_compressed(savename,hpf=np.real(recon))
    
def test_filt(filename,savename):
    dat=get_data(filename)
    dat=dat.astype('float')
    transformed = fft2(dat)
    hp = highpass(transformed)
    recon = ifft2(hp)
    np.savez_compressed(savename,hpfiltered=np.real(recon))
    
location='F:/BetaPic/acs/raw/'
prefix='hst_9861_'
numlist=['01','04','05','06','07','08','09','10']
suffix='_acs_wfc_f'
wvlist=['435','606','814']
suffix2='w_drz.fits'

pairs=[(0,1),(0,2),(1,2),(0,),(2,),(1,),(0,2),(1,2)]

if __name__ == '__main__':
    for i,n in enumerate(numlist):
        for j in pairs[i]:
            fname = location+n+'/'+prefix+n+suffix+wvlist[j]+suffix2
            sname = location+n+'/filtered'+n+'_'+wvlist[j]
            prep_filt(fname,sname)