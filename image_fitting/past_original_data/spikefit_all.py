import numpy as np
from astropy.io import fits
from skimage.transform import radon
from skimage.transform._warps_cy import _warp_fast
from scipy.ndimage import map_coordinates,gaussian_filter,median_filter
from scipy.optimize import curve_fit,root
import pdb
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from bad_pixel_fix import mask_bad_pix,fill_bad_pix

def line(x,m,b):
    return m*x+b
    
def subline(x,m1,b1,m2,b2):
    return line(x,m1,b1)-line(x,m2,b2)
    
def parab(x,a,b,c):
    return a*x**2 + b*x + c

def subparab(x,a1,b1,c1,a2,b2,c2):
    return parab(x,a1,b1,c1)-parab(x,a2,b2,c2)
    
def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d
    
def subcubic(x,a1,b1,c1,d1,a2,b2,c2,d2):
    return cubic(x,a1,b1,c1,d1)-cubic(x,a2,b2,c2,d2)
    
def crossparab(xarr,a,b,c,d,e,f):
    x=xarr[0]
    y=xarr[1]
    fy=parab(x,a,b,c)
    fx=parab(y,d,e,f)
    return np.array([x-fx,y-fy])
    
def crossline(xarr,m,b,n,c):
    x=xarr[0]
    y=xarr[1]
    fy=line(x,m,b)
    fx=line(y,n,c)
    return np.array([x-fx,y-fy])
    
def crosscubic(xarr,a,b,c,d,e,f,g,h):
    x=xarr[0]
    y=xarr[1]
    fy=cubic(x,a,b,c,d)
    fx=cubic(y,e,f,g,h)
    return np.array([x-fx,y-fy])


#------------------------------------------------------------------------------        
# Functions for preparing/shaping image to be fit
#------------------------------------------------------------------------------
def padding(img):
    '''
    Make a square image slightly more than sqrt(2) larger by zero-padding on all sides.
    Allows the original image to be rotated without any risk of data "falling off" the
    edge of the image.
    Returns the embiggened image, and the amount of zero-padding on the bottom and left.
    '''
    shp = img.shape[0]
    bigshp = np.ceil(shp*np.sqrt(2.))
    pad = (bigshp-shp)//2
    big = np.zeros((bigshp,bigshp))
    big[pad:pad+shp,pad:pad+shp]=img
    return big,pad
    
def derotate(img,drzangle):
    '''
    drzangle is recovered from headers
    and gives amt by which the original, diagonal-spike img
    was rotated to give a north-up image.
    Function will de-rotate by this amount, then rotate pi/4 to give vert & horiz spikes
    Note that giving drzangle = 0 will still rotate by pi/4.
    Since most non-north-up images have exactly diagonal spikes, this is perfect.
    '''
    derot_angle = -drzangle+np.pi/4.
    cen = img.shape[0]//2
    rot = _warp_fast(img,rotator(derot_angle,cen),order=2)
    return rot
    
def guess_center(img):
    '''
    Makes an educated guess at the image center in the following way:
    Sum image along rows & median filter resulting column vector.
    Max value of this column is roughly y value of img center.
    Sum along columns & filter resulting row vector to guess x value.
    
    In this first guess the flux from the star likely dominates the guess.
    
    Repeat this procedure, with a 200x200px square masked out around
    the (x,y) center just guessed. Return the new guess.
    
    This procedure (hopefully) minimizes the contribution to the guess
    from the pixels near the star center and maximizes the contribution
    from the diffraction spikes themselves.
    '''
    temp = np.nan_to_num(img.copy())
    xsum = np.sum(temp,axis=0) #collapse over 0th axis (i.e. stack rows, preserve columns)
    pixels = np.arange(float(temp.shape[0]))
    ysum = np.sum(temp,axis=1) #collapse over 1st axis (stack columns, preserve rows)
    f_xsum = median_filter(xsum,size=10)
    f_ysum = median_filter(ysum,size=10)
    mu_x = list(f_xsum).index(np.max(f_xsum))
    mu_y = list(f_ysum).index(np.max(f_ysum))
    sig = 20
    Ax = np.sum(f_xsum)
    Ay = np.sum(f_ysum)
    p0x = (mu_x,sig,Ax)
    p0y = (mu_y,sig,Ay)
    px,cx = curve_fit(gauss,pixels,f_xsum,p0=p0x)
    py,cy = curve_fit(gauss,pixels,f_ysum,p0=p0y)
    
    fg = (px[0],py[0])
    temp[fg[1]-100:fg[1]+100,fg[0]-100:fg[0]+100]=0.
    
    xsum = np.sum(temp,axis=0) #collapse over 0th axis (i.e. stack rows, preserve columns)
    pixels = np.arange(float(temp.shape[0]))
    ysum = np.sum(temp,axis=1) #collapse over 1st axis (stack columns, preserve rows)
    f_xsum = median_filter(xsum,size=10)
    f_ysum = median_filter(ysum,size=10)
    mu_x = list(f_xsum).index(np.max(f_xsum))
    mu_y = list(f_ysum).index(np.max(f_ysum))
    sig = 20
    Ax = np.sum(f_xsum)
    Ay = np.sum(f_ysum)
    p0x = (mu_x,sig,Ax)
    p0y = (mu_y,sig,Ay)
    px,cx = curve_fit(gauss,pixels,f_xsum,p0=p0x)
    py,cy = curve_fit(gauss,pixels,f_ysum,p0=p0y)
    
    return (px[0],py[0])

def rotator(theta,center):
    '''
    Build a rotation matrix to apply to the image via _warp_fast.
    Rotation will be counter-clockwise by 'theta' radians, centered on 'center.'
    '''
    shift0=np.array([[1,0,-center],[0,1,-center],[0,0,1]])
    shift1=np.array([[1,0,center],[0,1,center],[0,0,1]])
    R=np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
    return shift1.dot(R).dot(shift0)

def get_data(filename,index=1,drz=True):
    '''
    Retrieve data from fits file 'filename'.
    astropy.fits opens a list of hdu items; 'index' is used to retrieve data
    from the correct hdu in the list.
    'drz' defaults to True so that the rotation angle in drizzled data can be retrieved.
    It can be set to False if data has not been previously rotated; then
    images need only be rotated by pi/4 to have spikes vertical and horizontal.
    '''
    hdulist=fits.open(filename)
    rawdata=hdulist[index].data
    if drz:
        drzangle = np.radians(hdulist[0].header['D001ROT'])
    else:
        drzangle = 0.
    hdulist.close()
    return rawdata,drzangle
    
def gauss(x,A,mu,sig,o):
    '''
    gaussian with width sig, mean mu, normalization A and additive offset o
    '''
    norm = A/np.sqrt(2.*np.pi)/sig
    expon = ((x-mu)**2.)/(2.*sig**2.)
    return norm*np.exp(-expon)

def sinc2(x,a,p,s,o):
    '''
    np.sinc(x) = sin(pi*x)/(pi*x)
    this sinc2 fn uses 's' as a multiplicative width (small s = wider),
    'p' as a left-right offset, 'a' as a multiplicative prefactor.
    These values are then squared, and a vertical offset 'o' added.
    '''
    f=a*np.sinc(s*(x-p))
    return f**2+o
    

def twosinc2(x,a1,p1,s1,a2,p2,s2,o):
    '''
    Additive function with two sinc^2 profiles for doubled spikes (ie WFPC2)
    Contains only a single offset term to prevent degeneracy
    '''
    return sinc2(x,a1,p1,s1,o)+sinc2(x,a2,p2,s2,o)
    
def threesinc2(x,a1,p1,s2,a2,p2,s2,a3,p3,s3,o):
    '''
    Additive function with 3 sinc^2 profiles for doubled spikes (ie WFPC2)
    Contains only a single offset term to prevent degeneracy
    '''
    return sinc2(x,a1,p1,s1,o)+sinc2(x,a2,p2,s2,o)+sinc2(x,a3,p3,s3,o)

def build_par_guess(fitcoords,fitvals,cenguess,spikes,shp,i,vert=False):
    amp = np.sqrt(fitvals.max())
    s = 0.2
    o = 0.
    if vert: cen = cenguess[0]
    else: cen = cenguess[1]
    if spikes==1: p = (amp,cen,s,o)
    elif spikes==2:
        mid = shp//2
        imod = i-mid
        #spikes spread over the course of the image
        cen1 = cen-imod/100.
        cen2 = cen+imod/100.
        p1 = (amp,cen1,s)
        p2 = (amp,cen2,s,o)
        p = p1+p2
    else:
        hamp = amp/2.
        cen_up = cen+5.
        cen_dn = cen-5.
        p1 = (amp,cen,s)
        p2 = (hamp,cen_up,s)
        p3 = (hamp,cen_dn,s,o)
        p = p1+p2+p3
    return p
        
def pad_params(p):
    num_p = len(p)
    params = np.zeros((10.))
    params[:num_p-2] = p[:-1]
    params[-1]=p[-1]
    return params
    
def polynomial_fit(hinds,horiz,vinds,vert):
    #inds = np.arange(len(horiz))
    fith = horiz
    fitv = vert
    i_h = hinds
    i_v = vinds
    #i_h = i_h[fith<max(inds)]
    #i_v = i_v[fitv<max(inds)]
    #fith = fith[fith<max(inds)]
    #fitv = fitv[fitv<max(inds)]
    medh = np.median(fith) #median in case any remaining stragglers would throw off the mean
    medv = np.median(fitv)
    
    sigh = np.abs(fith-medh)+.1
    sigv = np.abs(fitv-medv)+.1
    #plt.plot(i_h,fith,'bo',fitv,i_v,'go')
    #plt.xlim([min(inds),max(inds)])
    #plt.ylim([min(inds),max(inds)])
    #plt.show()
    print len(i_h),len(fith),len(sigh)
    plh,c = curve_fit(line,i_h,fith,sigma=sigh)
    plv,c = curve_fit(line,i_v,fitv,sigma=sigv)
    pph,c = curve_fit(parab,i_h,fith,sigma=sigh)
    ppv,c = curve_fit(parab,i_v,fitv,sigma=sigv)
    pch,c = curve_fit(cubic,i_h,fith,sigma=sigh)
    pcv,c = curve_fit(cubic,i_v,fitv,sigma=sigv)
    
    return plh,plv,pph,ppv,pch,pcv
    
def find_ints(fn,guess,params):
    #pdb.set_trace()
    soln=root(fn,guess,args=params)
    if soln.success:
        return soln.x[0],soln.x[1]
    else:
        return -1,-1

def redchisquared(x,y,yfunc,params,err=None):
    if not err:
        err = np.ones(len(x))*np.std(y-yfunc(x,*params))
    dof = len(x)-len(params)-1
    if dof<=0: print "WARNING: Drastically overfitting data, no degrees of freedom left"
    return np.sum(((y-yfunc(x,*params))/err)**2.)/dof

def find_goods(rotated,horiz,vert,cenguess,iterations=2):
    '''
    Function for thinning down the fit data to only the successfully fit points.
    Take in a square image, the spike data for the horizontal and vertical spikes,
    and a guess as to the image center.
    
    Return, in order:
    good_hs : indices of well-fit horizontal spike points
    good_horiz : values of well-fit horizontal spike points
    good_vs : indices of well-fit vertical spike points
    good_vert : values of well-fit vertical spike points
    '''
    
    #arrays for indices/marking bad data
    bad_hs = []
    bad_vs = []
    
    shp = rotated.shape
    inds = np.arange(shp[0])
    
    
    #mark bad data: off image, or in a zeroed part of image
    for ind in inds:
        o_h = np.round(horiz[ind])
        o_v = np.round(vert[ind])
        if o_h>=shp[0]: bad_hs.append(ind)
        elif o_h<0: bad_hs.append(ind)
        elif rotated[o_h,ind]<=0: bad_hs.append(ind)
        
        if o_v>=shp[1]: bad_vs.append(ind)
        elif o_v<0: bad_vs.append(ind)
        elif rotated[ind,o_v]<=0: bad_vs.append(ind)
    #mask center
    for ind in range(int(cenguess[0])-100,int(cenguess[0])+100):
        if ind not in bad_hs: bad_hs.append(ind)
    for ind in range(int(cenguess[1])-100,int(cenguess[1])+100):
        if ind not in bad_vs: bad_vs.append(ind)
    
    #turn bad lists into good lists
    temp = list(inds.copy())
    for ind in temp[:]:
        if ind in bad_hs: temp.remove(ind)
    good_hs = temp[:]
    temp = list(inds.copy())
    for ind in temp[:]:
        if ind in bad_vs: temp.remove(ind)
    good_vs = temp[:]
    
    good_hs = np.array(good_hs)
    good_vs = np.array(good_vs)
    
    #pdb.set_trace()
    
    good_horiz = np.array(horiz[good_hs])
    good_vert = np.array(vert[good_vs])
    
    if len(good_hs)>300:
        good_hs = good_hs[30:-30]
        good_horiz = good_horiz[30:-30]
    elif len(good_hs)>150:
        good_hs = good_hs[15:-15]
        good_horiz = good_horiz[15:-15]
    elif len(good_hs)>30:
        good_hs = good_hs[5:-5]
        good_horiz = good_horiz[5:-5]
    if len(good_vs)>300:
        good_vs = good_vs[30:-30]
        good_vert = good_vert[30:-30]
    elif len(good_vs)>150:
        good_vs = good_vs[15:-15]
        good_vert = good_vert[15:-15]
    elif len(good_vs)>30:
        good_vs = good_vs[5:-5]
        good_vert = good_vert[5:-5]
        
    #Initial fit
    plh,c = curve_fit(line,good_hs,good_horiz)
    plv,c = curve_fit(line,good_vs,good_vert)
    
    print 'Chi squareds:'
    
    r1 = redchisquared(good_hs,good_horiz,line,plh)
    r2 = redchisquared(good_vs,good_vert,line,plv)
    print r1, r2
    
    for i in range(iterations):
        resid_h = good_horiz - line(good_hs,*plh)
        resid_v = good_vert - line(good_vs,*plv)
        where_small_h = np.where(np.abs(resid_h)<bad_deviation*np.std(resid_h))[0]
        where_small_v = np.where(np.abs(resid_v)<bad_deviation*np.std(resid_v))[0]
        
        good_hs = good_hs[where_small_h]
        good_vs = good_vs[where_small_v]
        good_horiz = good_horiz[where_small_h]
        good_vert = good_vert[where_small_v]
        
        r1 = redchisquared(good_hs,good_horiz,line,plh)
        r2 = redchisquared(good_vs,good_vert,line,plv)
        print r1, r2
        
        plh,c = curve_fit(line,good_hs,good_horiz)
        plv,c = curve_fit(line,good_vs,good_vert)
        
        r1 = redchisquared(good_hs,good_horiz,line,plh)
        r2 = redchisquared(good_vs,good_vert,line,plv)
        print r1, r2
        
    return good_hs,good_horiz,good_vs,good_vert

def check_params(px,p0):
    if (px[0]<0.85*p0[0]) or (px[0]>1.15*p0[0]): raise
    if px[2]<0.05: raise
    return

def fit_both(img,cenguess,spikes=1):
    '''
    Fitting function. Takes as argument an image w/ horizontal & vertical diffraction spikes
    and an initial guess at the star's location. Returns a list of threesinc2 parameters
    (if fewer than 3 spikes are present, the parameters for nonexistent spikes will be 0).
    Also returns a list of problematic list indices.
    '''
    xguess = cenguess[0]
    yguess = cenguess[1]
    shp = img.shape[0]
    cen=shp//2
    inds = np.arange(shp)
    px = np.zeros((10.,shp))-1.
    py = np.zeros((10.,shp))-1.
    badsx=[]
    badsy=[]
    if spikes==1:
        fitfn = sinc2
    elif spikes==2:
        fitfn = twosinc2
    elif spikes==3:
        fitfn = threesinc2
    else:
        print 'Must set spikes = 1, 2, or 3'
        raise ValueError
    if np.min(img)<0:
        img+= -np.min(img)
        img+= 0.001*np.mean(img)
    
    
    for i in inds:
        nz = np.where(img[:,i]>0)[0]
        imod = i-cen
        #offset_guess = 0.5*(np.min(img[:,i])+np.median(img[:,i]))
        try:
            if len(nz)>84:
                fitcoords = np.arange(int(round(yguess))-40,int(round(yguess))+40,1)
            elif len(nz)>4:
                fitcoords = nz[2:-2]
            else:
                fitcoords = inds
            fitvals = img[fitcoords,i]
            p0 = build_par_guess(fitcoords,fitvals,cenguess,spikes,shp,i)
            p,c = curve_fit(fitfn,fitcoords,fitvals,p0=p0,sigma=1./np.sqrt(np.abs(fitvals)),absolute_sigma=False)
            px[:,i] = pad_params(p)
        except:
            badsx.append(i)
            
    for i in inds:
        nz = np.where(img[i,:]>0)[0]
        imod = i-cen
        try:
            if len(nz)>84:
                fitcoords = np.arange(int(round(xguess))-40,int(round(xguess))+40,1)
            elif len(nz)>4:
                fitcoords = nz[2:-2]
            else:
                fitcoords = inds
            fitvals = img[i,fitcoords]
            p0 = build_par_guess(fitcoords,fitvals,cenguess,spikes,shp,i)
            p,c = curve_fit(fitfn,fitcoords,fitvals,p0=p0,sigma=1./np.sqrt(np.abs(fitvals)),absolute_sigma=False)
            py[:,i] = pad_params(p)