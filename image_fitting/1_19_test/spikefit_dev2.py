import numpy as np
from astropy.io import fits
from skimage.transform import radon
from skimage.transform._warps_cy import _warp_fast
from scipy.ndimage import map_coordinates,gaussian_filter
from scipy.optimize import curve_fit
import pdb

#the following command sequence prevents lag when drawing plots on windows.
#must be done in this order: need matplotlib but none of its sub-modules
#or sub-packages imported in order to call matplotlib.use()
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#A couple of these imports are unnecessary holdovers from when this code
#lived in the same module as radon transform code.


#Methods for fitting the uncooperative spike (and maybe improving the others):
#--Two-gaussian, flat offset
#--Three-gaussian
#--Two sinc^2, one gaussian
#--Three sinc^2
#
#--Offset to zero, fit whole thing (possibly better in the wings?)
#--Fit a small range around the spikes
#
#More esoteric:
#Danny's Gaussian processes (-.-)
#Some kind of MCMC model with my two- or three-Gaussian formulae
#Some kind of physically motivated correction: Based on the detailed properties of HST/WFC3
#
#Something iterative?
#--Find some metric for integrating up the residuals (like a chi^2)
#--Apply successively more Gaussians (starting from 2) to try to improve the fit
#--Integrate the residuals in specific segments; target the gaussians at problem segments
#         to improve convergence
#--Identify and correct different ways in which the fit fails; save these and generate
#         a table for automatic correction of common cases

'''LOOK HERE FIRST'''
#As of 1/19, fitting works--use twosinc2 and give the fitting algorithm VERY GOOD initial guesses,
#based on the table below.
#"TOP" and "BOT" spikes have been fit.
#TODO: Left and right. Data cleanup. Joint and individual linear and polynomial fits. Centroids.

#Spikes:      POS                                    NEG
#j70s: Top: 1600--1250 2400--1243&1262     Top: 1600--1190 2400--1185&1205
#      Bot: 900--1245 0--1245&1222         Bot: 900--1192 0--1187&1209
#      L: 900--1190 0--1185&1205           L: 900--1218 0--1240&1218 
#      R: 1600--1192  2400--1187&1209      R: 1600--1212 2400--1218&1200
#
#j71s: Top: 1600--1250 2400--1243&1262     Top: 1600--1191 2400--1185&1205
#      Bot: 900-1245 0--1245&1222          Bot: 900--1192 0--1186&1209
#      L: 900--1192 0--1185&1205           L: 900--1218 0--1218&1241
#      R: 1600--1192 2400--1186&1208       R: 1600: 1212 2400--1201&1218
#
#j72s: Top: 1600--1251 2400--1244&1261     Visual inspection confirms that negative-rotated image
#      Bot: 900--1245 0--1245&1222         Is identical to positively-rotated image, when properly rotated
#      L: 900--1191 0--1186&1205           #sosurprised #mypythoncommentsarehashtagsnow
#      R: 1600--1192 2400--1186&1208
#
#j73s: Top: 1600--1251 2400--1261&1244
#      Bot: 900--1245 0--1245&1222
#      L: 900--1190 0--1185&1205
#      R: 1600--1191 2400--1186&1208

def twosinc2(x,a1,a2,p1,p2,s1,s2):
    f1=a1*np.sinc(s1*(x-p1))
    f2=a2*np.sinc(s2*(x-p2))
    return f1**2 + f2**2


def fitscript(params,savefile,bounds):
    shp=params.shape[1]
    xdata=np.arange(shp)
    cen=float(shp//2)
    y1=params[2,:]
    y2=params[3,:]
    s1=params[4,:]
    s2=params[5,:]
    y1=y1[bounds[0]:bounds[1]]
    y2=y2[bounds[0]:bounds[1]]
    s1=s1[bounds[0]:bounds[1]]
    s2=s2[bounds[0]:bounds[1]]
    xdata=xdata[bounds[0]:bounds[1]]

    p1,c1=curve_fit(line,xdata,y1,p0=(0.,cen))
    p2,c2=curve_fit(line,xdata,y2,p0=(0.,cen))
    
    res1=y1-line(xdata,*p1)
    res2=y2-line(xdata,*p2)
    
    
    y1s=gaussian_filter(y1,20)
    y2s=gaussian_filter(y2,20)
    
    
    np.savez_compressed(savefile,x=x,y1=y1,y2=y2,r1=r1,r2=r2,p1=p1,p2=p2)
    p1,c1=curve_fit(line,xdata,y1s,p0=(0.,cen))
    p2,c2=curve_fit(line,xdata,y2s,p0=(0.,cen))
    np.savez_compressed(savefile+'smooth',p1=p1,p2=p2)
    



def testfnvert(img):
    shp=img.shape[0]
    inds=np.arange(shp)
    cen=shp//2
    c=cen+10
    
    badsx=[]
    badsy=[]
    
    paramsx=np.zeros((6.,shp))
    covsx=np.zeros((6.,6.,shp))
    paramsy=np.zeros((6.,shp))
    covsy=np.zeros((6.,6.,shp))
    altx=np.zeros(img.shape)
    alty=np.zeros(img.shape)
    
    for i in inds[inds>1500]:
        nz=np.where(img[i,:]>0)[0]
        imod=i-cen
        offset=img[i,nz]-np.mean(img[i,nz])
        p0=(1.3e2,1.3e2,c+25.*imod/float(shp),c-25.*imod/float(shp),3.,3.)
        try:
            paramsx[:,i],covsx[:,:,i]=curve_fit(twosinc2,nz[10:-10],offset[10:-10],p0=p0)
            altx[i,:]=twosinc2(inds,*paramsx[:,i])
        except:
            badsx.append(i)
        else:
            if paramsx[4,i]>10:
                badsx.append(i)
            elif paramsx[5,i]>10:
                badsx.append(i)
        print i
    return paramsx,covsx,altx,badsx
    
def choosefit(img,fitfn):
    shp=img.shape[0]
    inds=np.arange(shp)
    cen=shp//2
    c=cen-40
    
    badsx=[]
    badsy=[]
    
    paramsx=np.zeros((6.,shp))
    covsx=np.zeros((6.,6.,shp))
    paramsy=np.zeros((6.,shp))
    covsy=np.zeros((6.,6.,shp))
    altx=np.zeros(img.shape)
    alty=np.zeros(img.shape)
    
    for i in inds[inds>1500]:
        nz=np.where(img[:,i]>0)[0]
        imod=i-cen
        offset=img[nz,i]-np.mean(img[nz,i])
        #p0=(1.3e2,1.3e2,c+25.*imod/float(shp),c-25.*imod/float(shp),3.,3.)
        p0=(6,6,c+30.*imod/float(shp),c-10.*imod/float(shp),.15,.15)
        big=nz[np.where(offset>10)[0]]
        if len(big)>0:
            fitcoords=np.arange(np.min(big)-30,np.max(big)+30,1)
            fitvals=img[fitcoords,i]-np.mean(img[nz,i])
            try:
                paramsx[:,i],covsx[:,:,i]=curve_fit(fitfn,fitcoords,fitvals,p0=p0)
                altx[i,:]=fitfn(inds,*paramsx[:,i])
            except:
                badsx.append(i)
            else:
                if paramsx[4,i]>10:
                    badsx.append(i)
                elif paramsx[5,i]>10:
                    badsx.append(i)
        else:
            try:
                paramsx[:,i],covsx[:,:,i]=curve_fit(fitfn,nz[2:-2],offset[2:-2],p0=p0)
                altx[i,:]=fitfn(inds,*paramsx[:,i])
            except:
                badsx.append(i)
            else:
                if paramsx[4,i]>10:
                    badsx.append(i)
                elif paramsx[5,i]>10:
                    badsx.append(i)

        print i
    return paramsx,covsx,altx,badsx
    
def testfn(img):
    shp=img.shape[0]
    inds=np.arange(shp)
    cen=shp//2
    c=cen-40
    
    badsx=[]
    badsy=[]
    
    paramsx=np.zeros((6.,shp))
    covsx=np.zeros((6.,6.,shp))
    paramsy=np.zeros((6.,shp))
    covsy=np.zeros((6.,6.,shp))
    altx=np.zeros(img.shape)
    alty=np.zeros(img.shape)
    
    for i in inds[inds>1700]:
        nz=np.where(img[:,i]>0)[0]
        imod=i-cen
        offset=img[nz,i]-np.mean(img[nz,i])
        p0=(1.3e2,1.3e2,c+25.*imod/float(shp),c-25.*imod/float(shp),3.,3.)
        try:
            paramsx[:,i],covsx[:,:,i]=curve_fit(two_gauss,nz[10:-10],offset[10:-10],p0=p0)
            altx[:,i]=two_gauss(inds,*paramsx[:,i])
        except:
            badsx.append(i)
        else:
            if paramsx[4,i]>10:
                badsx.append(i)
            elif paramsx[5,i]>10:
                badsx.append(i)
        print i
    return paramsx,covsx,altx,badsx
    
def check(f):
    with np.load('posRT'+f+'.npz') as d:
        p=d['p']
    #with np.load('negTOP'+f+'.npz') as d:
    #    n=d['p']
    with np.load('rot'+f+'.fits.npz') as d:
        pos=d['pos']
    #    neg=d['neg']
    poscopy=pos.copy()
    poscopy[poscopy>250]=250
    poscopy[poscopy<150]=150
    #negcopy=neg.copy()
    #negcopy[negcopy>250]=250
    #negcopy[negcopy<150]=150
    plt.imshow(poscopy,origin='lower')
    inds=np.arange(0.,2462.,1.)
    plt.plot(inds,p[2,:],'bo',inds,p[3,:],'go')
    plt.show()
    #plt.imshow(negcopy,origin='lower')
    #plt.plot(n[2,:],inds,'bo',n[3,:],inds,'go')
    #plt.show()
    

#Fn to crop image to square, and mask outside of an inscribed circle
#Returns masked image & number of pixels removed from long dimension
#Works well
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

#Fn for making image rotation matrix for use in _warp_fast
#Works well bc someone else wrote it
def rotator(theta,center):
    shift0=np.array([[1,0,-center],[0,1,-center],[0,0,1]])
    shift1=np.array([[1,0,center],[0,1,center],[0,0,1]])
    R=np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
    return shift1.dot(R).dot(shift0)

fnamelist=['j70s','j71s','j72s','j73s']
#Main function; intended to fit diffraction spikes from scratch
#then plot to compare reconstructed images w/ original.
#Currently fitting spikes does not work correctly.
def routine(filename):
    with np.load('rot'+filename+'.fits.npz') as d:
        pos=d['pos']
        neg=d['neg']
    #try to fit (see fit_spikes)
    px,cx,ax,bx=choosefit(pos,twosinc2)
    
    #Save these parameters in case they're not worthless
    np.savez_compressed('posRT'+filename,p=px,c=cx,b=bx,a=ax)
    
    #py,cy,ay,by=choosefit(neg,twosinc2)
    
    #np.savez_compressed('negTOP'+filename,p=py,c=cy,b=by,a=ay)
    
    #Compare original w/ reconstructions
    #fig=plt.figure()
    #fig.add_subplot(211)
    #plt.imshow(rotated)
    #fig.add_subplot(212)
    #plt.imshow(ax)
    #plt.show()
    return
    
def prep_rot(filename):
    dat=get_data(filename)
    masked,remove=shape_image(dat) #make the image square & mask it
    shp=masked.shape[0]
    cen=shp//2
    rotated=_warp_fast(masked,rotator(np.pi/4.,cen))
    rot2=_warp_fast(masked,rotator(-np.pi/4.,cen))
    np.savez_compressed('rot'+filename,pos=rotated,neg=rot2)

#Fn to fit:
#Two gaussians on top of a "top hat" (truncated flat offset)
#Too many parameters to fit effectively?
#Better to do this by zeroing out the flat offset and only fitting 2 gaussians
@np.vectorize
def two_gauss(x,A1,A2,mu1,mu2,sigma1,sigma2):
    g1 = A1/sigma1/np.sqrt(2.*np.pi)*np.exp(-((x-mu1)**2.)/2./sigma1**2.)
    g2 = A2/sigma2/np.sqrt(2.*np.pi)*np.exp(-((x-mu2)**2.)/2./sigma2**2.)
    return g1+g2

    
def line(x,m,b):
    return m*x+b

#Ostensibly, fit diffraction spikes. Currently runs but returns crap.
def fit_spikes(image):
    #Image must be square
    assert image.shape[0]==image.shape[1]
    shp=image.shape[0]
    indices=np.arange(shp)
    
    #Lists of bad indices, should any occur
    badsx=[]
    badsy=[]
    
    #setup arrays to store fit parameters and covariances
    paramsx=np.zeros((9.,shp))
    covsx=np.zeros((9.,9.,shp))
    paramsy=np.zeros((9.,shp))
    covsy=np.zeros((9.,9.,shp))
    altx=np.zeros(image.shape)
    alty=np.zeros(image.shape)
    
    coords=np.arange(shp)
    cen=float(shp)/2.
    
    #Do horizontal spike first
    for i in indices:
        slice = image[:,i]
        slice = slice[slice>0]
        buffer=shp-len(slice)
        slice = image[:,i]
        p0=(1e3,1e3,cen+2.,cen-2.,4.,4.,150,buffer/2.,shp-buffer/2.)
        try:
            paramsx[:,i],covsx[:,:,i]=curve_fit(two_gauss,coords,slice,p0=p0)
            altx[:,i]=two_gauss(coords,*paramsx[:,i])
        except RuntimeError:
            badsx.append(i)
    
    #Then do vertical
    for i in indices:
        slice = image[i,:]
        slice = slice[slice>0]
        buffer=shp-len(slice)
        slice=image[i,:]
        p0=(1e3,1e3,cen+2.,cen-2.,4.,4.,150,buffer/2.,shp-buffer/2.)
        try:
            paramsy[:,i],covsy[:,:,i]=curve_fit(two_gauss,coords,slice,p0=p0)
            alty[i,:]=two_gauss(coords,*paramsy[:,i])
        except RuntimeError:
            badsy.append(i)
            
    #Return the following:
    #Fit parameters and covariances for the x & y slices
    #Bad indices (fit didn't converge) for x & y slices
    #Reconstructed images for x & y slices
    return paramsx,covsx,badsx,paramsy,covsy,badsy,altx,alty

#Fit a line to the maxima of the returned gaussians
#Indexing might not be up-to-date for this
#(since the returned gaussians are crap, I have not kept this fn in step with the other)
def line_fit(params,bad_indices=None):
    y1=params[2,:]
    y2=params[3,:]
    x=np.arange(float(params.shape[1]))
    sig1=params[4,:]
    sig2=params[5,:]
    line1=np.zeros((3,2))
    line2=np.zeros((3,2))
    
    if bad_indices is not None:
        sig1[bad_indices]=100000.
        sig2[bad_indices]=100000.
    
    line1[0,:],line1[1:,:]=curve_fit(line,x,y1,sigma=sig1)
    line2[0,:],line2[1:,:]=curve_fit(line,x,y2,sigma=sig2)
    plt.errorbar(x,y1,yerr=sig1,fmt='bo',ecolor='k',elinewidth=2,capthick=2,ms=2)
    plt.plot(x,y1,'bo')
    plt.errorbar(x,y2,yerr=sig2,fmt='ro',ecolor='k',elinewidth=2,capthick=2,ms=2)
    plt.plot(x,y2,'ro')
    plt.plot(x,line(x,*line1[0,:]),'b-')
    plt.plot(x,line(x,*line2[0,:]),'r-')
    plt.show()
    return line1,line2

    
#Fn to grab data
#Works well bc it's short
def get_data(filename):
    hdulist=fits.open(filename)
    rawdata=hdulist[0].data
    hdulist.close()
    return rawdata

#Silly stupid fn to trim image & record cuts made
#Removes outer 1/nth of image from each side
#Works correctly, but is not the best design for a "trim image" fn
def cut_data(rawdata,n=4):
    xcut=rawdata.shape[1]//n
    ycut=rawdata.shape[0]//n
    dat=rawdata[ycut:(n-1)*ycut,xcut:(n-1)*xcut]
    return dat,(xcut,ycut)
    
if __name__=='__main__':
    #for f in fnamelist:
    #    routine(f)
    for f in fnamelist:
        check(f)