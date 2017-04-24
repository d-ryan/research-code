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
#NOTE: changes have been made to this file to run on non-fmh stis images. 
#Dominic of the future:
#version control this garbage

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

#IMAGE ARRAYS ARE INDEXED [Y,X]

if __name__=='__main__':

    
    prefix = 'F:/BetaPic/wfc3/'
    bad_deviation = 1.5

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
        #if filename[-11:-9]=='wf':
        #elems=filename.split('_')
        #obsnum=int(filename[5:-7])
        if (filename[7]==2 or filename[7]==3): fnames.append(filename)
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
        px=np.zeros((4.,shp))-1.
        py=np.zeros((4.,shp))-1.
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
                p0=(fitvals.max(),yguess,.2,0.)
                px[:,i],c=curve_fit(sinc2,fitcoords,fitvals,p0=p0,sigma=1./np.sqrt(np.abs(fitvals)))
            except:
                badsx.append(i)
        elif len(nz)>4: #Not big enough? Fit everything, trimming off the edges
            try:
                p0=(offset[2:-2].max(),yguess,.2,0.)
                px[:,i],c=curve_fit(sinc2,nz[2:-2],offset[2:-2],p0=p0,sigma=1./np.sqrt(np.abs(offset[2:-2])))
            except:
                badsx.append(i)
        else: #Fit the whole thing; it's probably bad
            try:
                p0=(img[:,i].max(),yguess,.2,0.)
                px[:,i],c=curve_fit(sinc2,inds,img[:,i],p0=p0,sigma=1./np.sqrt(np.abs(img[:,i])))
            except:
                badsx.append(i)
                

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
                p0=(fitvals.max(),xguess,.2,0.)
                py[:,i],c=curve_fit(sinc2,fitcoords,fitvals,p0=p0,sigma=1./np.sqrt(np.abs(fitvals)))
            except:
                badsy.append(i)
        elif len(nz)>4: #Not big enough? Fit everything, trimming off the edges
            try:
                p0=(offset[2:-2].max(),xguess,.2,0.)
                py[:,i],c=curve_fit(sinc2,nz[2:-2],offset[2:-2],p0=p0,sigma=1./np.sqrt(np.abs(offset[2:-2])))
            except:
                badsy.append(i)
        else: #Fit the whole thing; it's probably bad
            try:
                p0=(img[i,:].max(),xguess,.2,0.)
                py[:,i],c=curve_fit(sinc2,inds,img[i,:],p0=p0,sigma=1./np.sqrt(np.abs(img[i,:])))
            except:
                badsy.append(i)
                

    return px,py,badsx,badsy


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
    elif xshp<yshp: #too tall, make wider
        new = np.zeros((yshp,yshp))
        diff = yshp-xshp
        new[:,diff//2:diff//2+xshp]=img.copy()
    else: #square
        new = img.copy()
        diff = 0.
    return new,diff

split = False
#Actual process loop
if __name__=='__main__':
    for i,f in enumerate(fnames):
        print '\nOperating on file',f

        img,drzangle = get_data(prefix+f,index=1,drz=False) #raw data
        square,diff = make_square(img)
        big,pad = padding(square) #padded; no rotation losses due to hitting edge of field
        rotated = derotate(big,drzangle) #rotated to have spikes vert & horiz
        
        cenguess = guess_center(rotated) #rough locations of said spikes
        print cenguess
        
        '''---NEEDS WORK---'''
        #Modify fit_both to accept a bool kwarg "split" for double spikes; default to False
        px,py,bx,by = fit_both(rotated,cenguess,split)
        
        #will need two sets of spike maxima
        horiz = px[1,:]
        vert = py[1,:]
        inds = np.arange(len(horiz))
        
        #Will need two sets for each direction
        good_hs,good_horiz,good_vs,good_vert = find_goods(rotated,horiz,vert,cenguess,iterations=5)
        
        #will need two sets for each direction
        plh,plv,pph,ppv,pch,pcv = polynomial_fit(good_hs,good_horiz,good_vs,good_vert)
        ltup = tuple(plh)+tuple(plv)
        ptup = tuple(pph)+tuple(ppv)
        ctup = tuple(pch)+tuple(pcv)
        
        #More intersects possible; not all are useful
        ints = np.zeros((3,2))
        
        ints[0,:] = find_ints(crossline,cenguess,ltup)
        ints[1,:] = find_ints(crossparab,cenguess,ptup)
        ints[2,:] = find_ints(crosscubic,cenguess,ctup)
        
        #savename = prefix+obsnums[i]+'/'+obsids[i]+'/'+obsnums[i]+'_'+obsids[i]+'_'+lambdas[i]
        savename = prefix+f
        snames.append(savename)
        angle=-drzangle+np.pi/4.
        np.savez_compressed(savename,img=rotated,orig=img,diff=diff,pad=pad,angle=angle,guess=cenguess,px=px,py=py,plh=plh,plv=plv,
            pph=pph,ppv=ppv,pch=pch,pcv=pcv,ints=ints,good_hs=good_hs,good_vs=good_vs,good_horiz=good_horiz,
            good_vert=good_vert,fname=f)
            
        plt.plot(inds,horiz,'bo',vert,inds,'go')
        plt.plot(inds,line(inds,*plh),'b-',line(inds,*plv),inds,'g-')
        plt.plot(inds,parab(inds,*pph),'b--',parab(inds,*ppv),inds,'g--')
        plt.plot(inds,cubic(inds,*pch),'b:',cubic(inds,*pcv),inds,'g:')
        plt.xlim([min(inds),max(inds)])
        plt.ylim([min(inds),max(inds)])
        plt.savefig(savename+'.svg')
        plt.clf()
        print 'Saved file: '+savename+'.npz'
        '''
        #ABOVE THIS POINT: Image has not been operated on.
        
        
        #BELOW THIS POINT: Bad-pixel-fixing.
        print 'BPF Image:'
        masked = mask_bad_pix(rotated)
        fixed = fill_bad_pix(masked)
        cenguess = guess_center(fixed) #rough locations of said spikes
        print cenguess
        px,py,bx,by = fit_both(fixed,cenguess)
        horiz = px[1,:]
        vert = py[1,:]
        inds = np.arange(len(horiz))
        
        plh,plv,pph,ppv,pch,pcv = polynomial_fit(horiz,vert)
        
        ltup = tuple(plh)+tuple(plv)
        ptup = tuple(pph)+tuple(ppv)
        ctup = tuple(pch)+tuple(pcv)
        
        ints = np.zeros((3,2))
        
        ints[0,:] = find_ints(crossline,cenguess,ltup)
        ints[1,:] = find_ints(crossparab,cenguess,ptup)
        ints[2,:] = find_ints(crosscubic,cenguess,ctup)
        
        savename = prefix+obsnums[i]+'/'+obsids[i]+'/'+obsnums[i]+'_'+obsids[i]+'_'+lambdas[i]+'_bpf'
        snames.append(savename)
        angle=-drzangle+np.pi/4.
        np.savez_compressed(savename,img=fixed,pad=pad,angle=angle,guess=cenguess,px=px,py=py,plh=plh,plv=plv,
            pph=pph,ppv=ppv,pch=pch,pcv=pcv,ints=ints,fname=f)
            
        plt.plot(inds,horiz,'bo',vert,inds,'go')
        plt.plot(inds,line(inds,*plh),'b-',line(inds,*plv),inds,'g-')
        plt.plot(inds,parab(inds,*pph),'b--',parab(inds,*ppv),inds,'g--')
        plt.plot(inds,cubic(inds,*pch),'b:',cubic(inds,*pcv),inds,'g:')
        plt.xlim([min(inds),max(inds)])
        plt.ylim([min(inds),max(inds)])
        plt.savefig(savename+'.svg')
        plt.clf()
        print 'Saved file: '+savename+'.npz'
        '''
        
        
        
        
        
        
        
        
        
        
        
        #    print 'Something went wrong for',f
        #    errors.append(f)
    np.savez_compressed(prefix+'info',snames=snames,errors=errors)
    #To do: Test this, then apply it to HPF, BPF, and smoothed images

'''
#Deprecated
def choosefit(img,sinc2):
    
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
                paramsx[:,i],covsx[:,:,i]=curve_fit(sinc2,fitcoords,fitvals,p0=p0,sigma=1./np.sqrt(np.abs(fitvals)))
                altx[:,i]=sinc2(inds,*paramsx[:,i])
            except:
                badsx.append(i)
            else:
                if paramsx[2,i]>10:
                    badsx.append(i)
        else:
            try:
                paramsx[:,i],covsx[:,:,i]=curve_fit(sinc2,nz[2:-2],offset[2:-2],p0=p0)
                altx[:,i]=sinc2(inds,*paramsx[:,i])
            except:
                badsx.append(i)
            else:
                if paramsx[2,i]>10:
                    badsx.append(i)

        print i
    return paramsx,covsx,altx,badsx


    
    
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
    for i,n in enumerate(numlist): #Image number
        for j in pairs[i]: #Wavelength
            #if i==2: break
            fname = location+n+'/fits_'+n+'_'+wvlist[j]+'.npz'

            with np.load(fname) as d:
                horiz=d['horiz']
                vert=d['vert']
                plh=d['plh']
                pph=d['pph']
                pch=d['pch']
                plv=d['plv']
                ppv=d['ppv']
                pcv=d['pcv']
            
            ints=np.zeros((3,2))
            ploth=np.arange(horiz[0].max())
            plotv=np.arange(vert[0].max())
            
            guess = (2000,2000)
            crosstupl=tuple(plh)+tuple(plv)
            soln=root(crossline,guess,args=crosstupl)
            if soln.success:
                ints[0,0]=soln.x[0]
                ints[0,1]=soln.x[1]
            crosstupp=tuple(pph)+tuple(ppv)
            soln=root(crossparab,guess,args=crosstupp)
            if soln.success:
                ints[1,0]=soln.x[0]
                ints[1,1]=soln.x[1]
            crosstupc=tuple(pch)+tuple(pcv)
            soln=root(crosscubic,guess,args=crosstupc)
            if soln.success:
                ints[2,0]=soln.x[0]
                ints[2,1]=soln.x[1]
                
            plt.plot(horiz[0],horiz[1],'bo',vert[1],vert[0],'go')
            plt.plot(ploth,line(ploth,*plh),'b-',line(plotv,*plv),plotv,'g-')
            plt.plot(ploth,parab(ploth,*pph),'b--',parab(plotv,*ppv),plotv,'g--')
            plt.plot(ploth,cubic(ploth,*pch),'b:',cubic(plotv,*pcv),plotv,'g:')
            plt.plot(ints[:,0],ints[:,1],'ro')
            #plt.show()
            plt.savefig(location+n+'/intersects_plot.pdf')
            plt.savefig(location+n+'/intersects_plot.svg')
            np.savez_compressed(location+n+'/intdata_'+n+'_'+wvlist[j],ints=ints)
            
            print n,wvlist[j]

            
            
if __name__=='__main__':
    for k,prefix in enumerate(prefixlist):
        for i,n in enumerate(numlist[k]): #Image number
            for j in pairs[k][i]: #Wavelength
                #if j<2: continue
                fname = location+prefix+'_'+n+suffix+wvlist[j]+suffix2
                sname = location+'/'+prefix+'/'+n+'/horiz'+'_'+wvlist[j]

                routine(fname,sname)
                check(fname,sname)
'''