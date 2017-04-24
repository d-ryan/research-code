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

location='F:/BetaPic/acs/raw/'
prefix='hst_9861_'
numlist=['01','04','05','06','07','08','09','10']
suffix='_acs_wfc_f'
wvlist=['435','606','814']
suffix2='w_drz.fits'

pairs=[(0,1),(0,2),(1,2),(0,),(2,),(1,),(0,2),(1,2)]

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
    
    for i in inds[inds>2820]:
        #if i>2200: break
        nz=np.where(img[:,i]>0)[0]
        imod=i-cen
        offset=img[nz,i]-np.mean(img[nz,i])
        #p0=(1.3e2,1.3e2,c+25.*imod/float(shp),c-25.*imod/float(shp),3.,3.)
        p0=(6,2825.,.15)
        big=nz[np.where(offset>10)[0]]
        if len(big)>0:
            fitcoords=np.arange(int(p0[1])-25,int(p0[1])+25,1)
            
            fitvals=img[fitcoords,i]-np.mean(img[nz,i])
            
            try:
                paramsx[:,i],covsx[:,:,i]=curve_fit(fitfn,fitcoords,fitvals,p0=p0)
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
    
def prep_rot(filename,savename,fnameold):
    #dat=get_data(filename)
    
    with np.load(filename+'.npz') as d:
        dat = d['hpf']
    
    masked,remove=shape_image(dat) #make the image square & mask it
    shp=masked.shape[0]
    cen=shp//2
    with np.load(fnameold+'.npz') as d:
        angle = d['angle']
    #angle=-np.arctan2(3131-2293,2128-2125)#-np.arctan2(1306-942,1092-909) #-(np.arctan2(-3173+3244,-16)-np.pi/2.)
    rotated=_warp_fast(masked,rotator(angle,cen))
    np.savez_compressed(savename,pos=rotated,angle=angle)
    
def sinc2(x,a,p,s):
    f=a*np.sinc(s*(x-p))
    return f**2
    
def shape_image(image):
    lowpix=np.min(image)
    image-=lowpix
    
    shp=image.shape
    xshp=shp[1]
    yshp=shp[0]
    
    xcen=xshp/2
    ycen=yshp/2
    
    #Make image square
    if xshp>yshp: #Image is too wide
        size=yshp
        remove=xshp-yshp
        if remove%2==0:
            cropped=image[:,remove/2:-remove/2]
        else:
            pre=np.ceil(float(remove)/2.)
            post=np.floor(float(remove)/2.)
            cropped=image[:,pre:-post]
    elif yshp>xshp: #Image is too tall
        size=xshp
        remove=yshp-xshp
        if remove%2==0:
            cropped=image[remove/2:-remove/2,:]
        else:
            pre=np.ceil(float(remove)/2.)
            post=np.floor(float(remove)/2.)
            cropped=image[pre:-post,:]
    else: #Image is already square
        size=xshp
        cropped=image.copy()
        remove=0.
    radius=size/2
    xx,yy=np.meshgrid(np.arange(size),np.arange(size))
    mask=np.ones((size,size))
    if size%2==0:
        mask[np.sqrt((xx+0.5-radius)**2+(yy+0.5-radius)**2)>=radius]=0
    else:
        mask[np.sqrt((xx-radius)**2+(yy-radius)**2)>radius]=0
    masked=cropped*mask
    return masked,remove
    
def routine(filename,savename):
    with np.load(filename+'.npz') as d:
        pos=d['pos']

    px,cx,ax,bx=choosefit(pos,sinc2)
    
    #Save these parameters in case they're not worthless
    np.savez_compressed(savename,p=px,c=cx,b=bx,a=ax)
    return
    
def check(rname,sname):
    with np.load(sname+'.npz') as d:
        p=d['p']
    #with np.load('negTOP'+f+'.npz') as d:
    #    n=d['p']
    with np.load(rname+'.npz') as d:
        pos=d['pos']
    #    neg=d['neg']
    poscopy=pos.copy()
    poscopy[poscopy>250]=250
    poscopy[poscopy<2]=2
    #negcopy=neg.copy()
    #negcopy[negcopy>250]=250
    #negcopy[negcopy<150]=150
    plt.imshow(poscopy,origin='lower')
    inds=np.arange(0.,float(pos.shape[0]),1.)
    plt.plot(inds,p[1,:],'bo')
    plt.show()

    
if __name__=='__main__':
    for i,n in enumerate(numlist):
        for j in pairs[i]:
            if i!=0: continue
            fnameold = location+n+'/rot'+prefix+n+suffix+wvlist[j]+suffix2
            fname = location+n+'/filtered'+n+'_'+wvlist[j]
            rname = location+n+'/filtrot'+n+'_'+wvlist[j]#prefix+n+suffix+wvlist[j]+suffix2
            sname = location+n+'/filtRIGHT'+n+'_'+wvlist[j]
            
            #prep_rot(fname,rname,fnameold)
            routine(rname,sname)
            check(rname,sname)