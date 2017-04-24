'''
New, complete spikefitting function

1--receive filename[s]; retain for later reference
2--load up the image, pad & derotate it
3--display the file[s] on log scale and request approximate center (should be ~1567,1061)
4--fit horizontal & vertical spikes
5--apply polynomial fits
6--calculate intersects
7--save all the data: spikes, fits, ints, original fname; save also fits & ints overlaid on image

8--repeat 4-7 for a high-pass-filtered image
9--repeat 4-7 for a bad-pixel-fixed image
10--repeat 4-7 for a smoothed image
11--HPF&BPF
12--HPF&smoothed
13--BPF&smoothed
14--HPF,BPF,smoothed

To-do: compare radon transform (from skimage & the more cost-effective method) vs this thing
       and evaluate accuracy/uncertainty of each centering method
       
       plug outlier-resistant fit into this thing
'''

import numpy as np
from astropy.io import fits
from skimage.transform import radon,probabilistic_hough_line
from skimage.feature import canny
from skimage.transform._warps_cy import _warp_fast
from scipy.ndimage import map_coordinates,gaussian_filter,median_filter
from scipy.optimize import curve_fit,root
import pdb
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from bad_pixel_fix import mask_bad_pix,fill_bad_pix
from fasttest import search_center
import time

#IMAGE ARRAYS ARE INDEXED [Y,X]

if __name__=='__main__':

    ######Beta-pic specific directory organization
    prefix = 'F:/sirius_astrometry/'
    #prefix = 'F:/stis_go13362/'
    bad_deviation = 2.5

    #First, look at all the files, make note of their names, make sure we have
    #directories to store them in
    fnames = []
    obsnums = []
    obsids = []
    lambdas = []
    snames = []
    errors = []

    #Note names & values
    for filename in os.listdir(prefix):
        #if (filename[-8:]=='flt.fits') or (filename[-8:]=='drz.fits'):
        if (filename[-8:]=='drz.fits'):
            #elems=filename.split('_')
            #obsnum=int(filename[5:-7])
            fnames.append(filename)
            #obsnums.append(filename[5:-7])
            #obsids.append(elems[2])
            #lambdas.append(elems[4])

    nfiles = len(fnames)        
    print 'Operating on '+str(nfiles)+' files.'
'''
#Make directories if needed
for i,id in enumerate(obsids):
    try:
        os.chdir(prefix+obsnums[i]+'/'+id)
        os.chdir(prefix)
    except WindowsError:
        os.mkdir(prefix+obsnums[i]+'/'+id)
        print 'Made new directory: '+obsnums[i]+'/'+id

print 'All needed directories present.'
'''
#prefix = 'F:/Fomalhaut/'
#os.chdir(prefix)
#fnames = ['vi01isu.fits']

def redchisquared(x,y,yfunc,params,err=None):
    if not err:
        err = np.ones(len(x))*np.std(y-yfunc(x,*params))
    dof = len(x)-len(params)-1
    if dof<=0: print "WARNING: Drastically overfitting data, no degrees of freedom left"
    return np.sum(((y-yfunc(x,*params))/err)**2.)/dof


#------------------------------------------------------------------------------        
# Functions for polynomial fits to entire spikes
#------------------------------------------------------------------------------
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
    #Sirius specific modifications
    flat = rawdata.sum(0)
    max=np.argmax(flat)
    rawdata[:,max-40:max+40]=0.
    return rawdata,drzangle
    
def gauss(x,mu,sig,A):
    '''
    gaussian with width sig, mean mu, multiplicative prefactor A
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
    
def tsinc2(x,a,p,s,o,w,d):
    return sinc2(x,a,p,s,o)+sinc2(x,a/d,p-w,s,0.)+sinc2(x,a/d,p+w,s,0.)
    
def threesinc2(x,a1,p1,s1,a2,p2,s2,a3,p3,s3,o):
    return sinc2(x,a1,p1,s1,o)+sinc2(x,a2,p2,s2,o)+sinc2(x,a3,p3,s3,o)
    
#------------------------------------------------------------------------------        
# Function that does the row-by-row & column-by-column fitting
#------------------------------------------------------------------------------
def fit_both(img,cenguess,split=False):
    '''
    fitting function. Takes as arguments an image with horizontal & vertical diffraction spikes
    and an initial guess at the star's location.
    
    Returns a list of sinc^2 parameters and a list of potentially problematic image indices
    for each of the two spikes.
    '''
    xguess = cenguess[0]
    yguess = cenguess[1]
    shp = img.shape[0]
    cen=shp//2
    inds = np.arange(shp)
    if split:
        px=np.zeros((7.,shp))-1.
        py=np.zeros((7.,shp))-1.
        badsx=[[],[]]
        badsy=[[],[]]
    else:
        px=np.zeros((6.,shp))-1.
        py=np.zeros((6.,shp))-1.
        px2=np.zeros((4.,shp))-1.
        py2=px2.copy()
        badsx=[]
        badsy=[]
    if np.min(img)<0:
        img=img.copy()
        img+=-np.min(img)+0.01
    #fit horizontal spike
    print 'Fitting horizontal'
    for i in inds: #loop over columns
        nz = np.where(img[:,i]>0)[0]
        imod = i-cen
        offset = img[nz,i]-np.mean(img[nz,i])
        #print i
        #bounds=([offset.max()/100,yguess-40,-np.inf,-np.inf],[offset.max()*100,yguess+40,np.inf,np.inf])
        if len(nz)>84: #Fit the central 80 pixels (arbitrary)
            fitcoords = np.arange(int(round(yguess))-40,int(round(yguess))+40,1)
            fitvals = img[fitcoords,i]-np.mean(img[nz,i])
            try:
                p0=(fitvals.max(),yguess,.2,0.,3.,2.)
                px[:,i],c=curve_fit(tsinc2,fitcoords,fitvals,p0=p0,sigma=1./np.sqrt(np.abs(fitvals)))
            except:
                badsx.append(i)
        elif len(nz)>4: #Not big enough? Fit everything, trimming off the edges
            try:
                p0=(offset[2:-2].max(),yguess,.2,0.,3.,2.)
                px[:,i],c=curve_fit(tsinc2,nz[2:-2],offset[2:-2],p0=p0,sigma=1./np.sqrt(np.abs(offset[2:-2])))
            except:
                badsx.append(i)
        else: #Fit the whole thing; it's probably bad
            try:
                p0=(img[:,i].max(),yguess,.2,0.,3.,2.)
                px[:,i],c=curve_fit(tsinc2,inds,img[:,i],p0=p0,sigma=1./np.sqrt(np.abs(img[:,i])))
            except:
                badsx.append(i)
        px2[0,i]=np.argmax(img[:,i])
        px2[1,i]=np.argmax(img[:px2[0,i]-1,i])
        px2[2,i]=np.argmax(img[px2[0,i]+2:,i])+px2[0,i]+2
        xfit = [px2[1,i],px2[0,i],px2[2,i]]
        yfit = [img[int(px2[1,i]),i],img[int(px2[0,i]),i],img[int(px2[2,i]),i]]
        #if np.any(np.isnan(xfit)) or np.any(np.isnan(yfit)):
        #    print i
        #    pdb.set_trace()
        par,c = curve_fit(parab,xfit,yfit)
        px2[3,i]=-par[1]/2./par[0]
                

    #fit vertical spike
    print 'Fitting vertical'
    for i in inds: #loop over columns
        nz = np.where(img[i,:]>0)[0]
        imod = i-cen
        offset = img[i,nz]-np.mean(img[i,nz])
        #bounds=([offset.max()/100,xguess-40,-np.inf,-np.inf],[offset.max()*100,xguess+40,np.inf,np.inf])
        if len(nz)>84: #Fit the central 80 pixels (arbitrary)
            fitcoords = np.arange(int(round(xguess))-40,int(round(xguess))+40,1)
            fitvals = img[i,fitcoords]-np.mean(img[i,nz])
            
            try:
                p0=(fitvals.max(),xguess,.2,0.,3.,2.)
                py[:,i],c=curve_fit(tsinc2,fitcoords,fitvals,p0=p0,sigma=1./np.sqrt(np.abs(fitvals)))
            except:
                badsy.append(i)
        elif len(nz)>4: #Not big enough? Fit everything, trimming off the edges
            try:
                p0=(offset[2:-2].max(),xguess,.2,0.,3.,2.)
                py[:,i],c=curve_fit(tsinc2,nz[2:-2],offset[2:-2],p0=p0,sigma=1./np.sqrt(np.abs(offset[2:-2])))
            except:
                badsy.append(i)
        else: #Fit the whole thing; it's probably bad
            try:
                p0=(img[i,:].max(),xguess,.2,0.,3.,2.)
                py[:,i],c=curve_fit(tsinc2,inds,img[i,:],p0=p0,sigma=1./np.sqrt(np.abs(img[i,:])))
            except:
                badsy.append(i)
        py2[0,i]=np.argmax(img[i,:])
        py2[1,i]=np.argmax(img[i,:py2[0,i]-1])
        py2[2,i]=np.argmax(img[i,py2[0,i]+2:])+py2[0,i]+2
        xfit = [py2[1,i],py2[0,i],py2[2,i]]
        yfit = [img[i,int(py2[1,i])],img[i,int(py2[0,i])],img[i,int(py2[2,i])]]
        par,c = curve_fit(parab,xfit,yfit)
        py2[3,i]=-par[1]/2./par[0]
                

    return px,py,badsx,badsy,px2,py2


def polynomial_fit(hinds,horiz,vinds,vert):
    
    fith = horiz
    fitv = vert
    i_h = hinds
    i_v = vinds
    
    plh,c = curve_fit(line,i_h,fith)#,sigma=sigh)
    plv,c = curve_fit(line,i_v,fitv)#,sigma=sigv)
    
    
    
    pph,c = curve_fit(parab,i_h,fith)#,sigma=sigh)
    ppv,c = curve_fit(parab,i_v,fitv)#,sigma=sigv)
    pch,c = curve_fit(cubic,i_h,fith)#,sigma=sigh)
    pcv,c = curve_fit(cubic,i_v,fitv)#,sigma=sigv)
    
    return plh,plv,pph,ppv,pch,pcv
    
def find_ints(fn,guess,params):
    soln=root(fn,guess,args=params)
    if soln.success:
        return soln.x[0],soln.x[1]
    else:
        return -1,-1


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
    
def make_square(img):
    '''
    Take in an MxN image. Return an MxM or NxN image, whichever is larger, by zero-padding
    the original image. if M==N, return original image.
    '''
    shp = img.shape
    xshp = shp[1]
    yshp = shp[0]
    if xshp>yshp: #too wide, make taller
        new = np.zeros((xshp,xshp))
        diff = xshp-yshp
        new[diff//2:diff//2+yshp,:]=img.copy()
        flag='y'
    elif xshp<yshp: #too tall, make wider
        new = np.zeros((yshp,yshp))
        diff = yshp-xshp
        new[:,diff//2:diff//2+xshp]=img.copy()
        flag='x'
    else: #square
        new = img.copy()
        diff = 0.
        flag=None
    return new,diff,flag

split = False
#Actual process loop
if __name__=='__main__':
    for i,f in enumerate(fnames):
        #if i>6: break
        try:
            start = time.clock()
            print '\nOperating on file',f
            img,drzangle = get_data(prefix+f,index=1,drz=False)
            img[np.isnan(img)]=np.nanmin(img)
            
            collx = np.sum(img,axis=1)
            colly = np.sum(img,axis=0)
            
            collx_f = median_filter(collx,size=100)
            colly_f = median_filter(colly,size=100)
            
            ymax = np.argmax(collx_f)
            xmax = np.argmax(colly_f)
            flat = img.sum(0)
            max=np.argmax(flat)
            img[:,max-40:max+40]=0.
            
            edges = canny(img,sigma=5)
            lines = probabilistic_hough_line(edges,threshold=150,line_length=100)
            
            up_flag = False
            dn_flag = False
            for lin in lines:
                p0,p1 = lin
                if p1[0]==p0[0]: continue
                if p1[1]==p0[1]: continue
                p0 = (float(p0[0]),float(p0[1]))
                p1 = (float(p1[0]),float(p1[1]))
                m = (p1[1]-p0[1])/(p1[0]-p0[0])
                if (((m<1.15) and (m>.85)) and not up_flag):
                    m_up = m
                    b_up = p1[1]-m_up*p1[0]
                    up = lin
                    up_flag = True
                if (((m > -1.15) and (m < -.85)) and not dn_flag):
                    m_dn = m
                    b_dn = p1[1]-m_dn*p1[0]
                    dn = lin
                    dn_flag = True
                if up_flag and dn_flag: break
            
            tup = tuple((m_up,b_up))+tuple((m_dn,b_dn))
            
            h_int = find_ints(crossline,(img.shape[0]/2.,img.shape[1]/2.),tup)
            
            angle_dn = np.arctan(m_dn)
            angle_up = np.arctan(m_up)

            square,diff,flag = make_square(img)
            big,pad = padding(square) #padded; no rotation losses due to hitting edge of field
            rotated = derotate(big,np.pi/4.-angle_dn) #rotated to have spikes vert & horiz
            rotated[np.isnan(rotated)]=np.nanmin(rotated)
            cenguess = guess_center(rotated) #rough locations of said spikes
            print cenguess
            savename = prefix+'radon/'+f
            rcen0 = search_center(rotated,cenguess[0],cenguess[1],rotated.shape[0]/2.,size_cost=60,theta=[0,90],decimals=0,save=savename+'_rot_search',m=0.4,M=0.6)
            rcen = search_center(rotated,rcen0[0],rcen0[1],rotated.shape[0]/2.,size_cost=10,theta=[0,90],decimals=2,save=savename+'_rot')
            print 'Rotated found'
            orig_cen0 = search_center(img,h_int[0],h_int[1],img.shape[0]/2.,size_cost=60,theta=[np.degrees(angle_up),np.degrees(angle_dn)+180],decimals=0,save=savename+'_orig_search',m=0.4,M=0.6)
            orig_cen = search_center(img,orig_cen0[0],orig_cen0[1],img.shape[0]/2.,size_cost=10,theta=[np.degrees(angle_up),np.degrees(angle_dn)+180],decimals=2,save=savename+'_orig')
            print 'Original found'
            snames.append(savename)
            angle=angle_dn
            np.savez_compressed(savename,img=rotated,orig=img,diff=diff,pad=pad,angle=angle,guess=cenguess,
                rcen=rcen,h_int=h_int,orig_cen=orig_cen,up=up,dn=dn,fname=f)
            
            plt.imshow(np.log(img),origin='lower',cmap=plt.cm.gray)
            plt.plot(orig_cen[0],orig_cen[1],'bo')
            plt.plot(h_int[0],h_int[1],'go')
            #plt.plot(xmax,ymax,'ro')
            plt.savefig(savename+'_orig.svg')
            plt.clf()
            plt.imshow(np.log(rotated),origin='lower',cmap=plt.cm.gray)
            plt.plot(cenguess[0],cenguess[1],'go')
            plt.plot(rcen[0],rcen[1],'bo')
            plt.savefig(savename+'_rot.svg')
            plt.clf()
            print 'Saved files: '+savename
            print 'Elapsed time: ',time.clock()-start

        except:
            print 'Something went wrong for',f
            errors.append(f)
    np.savez_compressed(prefix+'info',snames=snames,errors=errors)
    #To do: Test this, then apply it to HPF, BPF, and smoothed images
