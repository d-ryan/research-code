'''
Functions to fix:
guess_center
get_data - generalizing; if called incorrectly, should ask user to fix problems rather than ending program
check_params - comparison against quantitative values for fn means, amplitudes, std devs
find_goods - Pass in covariance matrix (or just param variance) and "bad" flags from fitting spike slices
             Should rely less on linear fits to spike data : perhaps try to minimize residuals (squared?) of diff. polynomial fit orders
fit_both - Rewrite after others improved
'''

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

#------------------------------------------------------------------------------        
# Functions for find_ints to compute intersects of various polynomial fit orders
#------------------------------------------------------------------------------
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

def polynomial_fit(hinds,horiz,vinds,vert):
    #inds = np.arange(len(horiz))
    fith = horiz
    fitv = vert
    i_h = hinds
    i_v = vinds
    
    plh,c = curve_fit(line,i_h,fith)
    plv,c = curve_fit(line,i_v,fitv)
    pph,c = curve_fit(parab,i_h,fith)
    ppv,c = curve_fit(parab,i_v,fitv)
    pch,c = curve_fit(cubic,i_h,fith)
    pcv,c = curve_fit(cubic,i_v,fitv)
    
    return plh,plv,pph,ppv,pch,pcv
    
def find_ints(fn,guess,params):
    #pdb.set_trace()
    soln=root(fn,guess,args=params)
    if soln.success:
        return soln.x[0],soln.x[1]
    else:
        return -1,-1

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
    bigshp = int(np.ceil(shp*np.sqrt(2.)))
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

def rotator(theta,center):
    '''
    Build a rotation matrix to apply to the image via _warp_fast.
    Rotation will be counter-clockwise by 'theta' radians, centered on 'center.'
    '''
    shift0=np.array([[1,0,-center],[0,1,-center],[0,0,1]])
    shift1=np.array([[1,0,center],[0,1,center],[0,0,1]])
    R=np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
    return shift1.dot(R).dot(shift0)

'''
NOTE: Find a better method of this
--coords of brightest pixel in image?
--coords of brightest pixel in image slice, fit line through?
--fitting gaussian to smoothed image slices
--Repeat 3 above, but with center masked out
'''
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
    w = np.argmax(temp)
    #For a 2D array, np.argmax returns the coordinate in the flattened array
    #The original array coords are recoverable by:
    yshp = temp.shape[1]
    img_max_coord=(w//yshp,w%yshp)
    img_max = temp[img_max_coord[0],img_max_coord[1]]

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
    
    return (px[1],py[1])

'''
NOTE: This needs to be more general, and there need to be external if-else clauses
to adjust which instrument the data are coming from.

Visually inspect ACS, WFC3, STIS datasets and search for rotation keywords/FITS header structure.
'''
def get_data(filename,index=None,drz=True):
    '''
    Retrieve data from fits file 'filename'.
    astropy.fits opens a list of hdu items; 'index' is used to retrieve data
    from the correct hdu in the list.
    'drz' defaults to True so that the rotation angle in drizzled data can be retrieved.
    It can be set to False if data has not been previously rotated; then
    images need only be rotated by pi/4 to have spikes vertical and horizontal.
    '''
    hdulist=fits.open(filename)

    if index is not None:
        try:
            rawdata=hdulist[index].data
        except IndexError:
            
    else:
        for i,hdu in enumerate(hdulist):
            
    
    if drz:
        drzangle = np.radians(hdulist[0].header['D001ROT'])
    else:
        drzangle = 0.
    hdulist.close()
    return rawdata,drzangle

def auto_retrieve(hdulist):
    max_shp = 0
    for i,hdu in enumerate(hdulist):
        data = hdu.data
        header = hdu.header
        if np.max(data.shape)>max_shp:
            max_shp = np.max(data.shape)
            index = i
    keys = header.keys()
    
    if 'TELESCOP' in keys:
        print header['TELESCOP']
    if 'INSTRUME' in keys:
        inst = header['INSTRUME']

    if 'RA_TARG' in keys: #RA of target
    if 'DEC_TARG' in keys: #Dec of target
    if 'PA_V3' in keys:
        pa_v3 = header['PA_V3']
        #This is the angle of the HST V3 axis
        #One diffraction spike is roughly parallel to this angle
        #(De)rotating by this amt should therefore eliminate problems
    'SUBARRAY' # T if from a subarray, F if full frame
    'DETECTOR' #Ex: WFC, HRC, or SBC (ACS will always be WFC); WFC3 IR vs UVIS
    'FILTER1' #Filter wheel 1 element
    'FILTER2' #Filter wheel 2 element
    'WCSAXES' #Number of World Coordinate System axes
    'CRPIX1' #x coord of reference pixel
    'CRPIX2' #y coord of reference pixel
    'CRVAL1' #First axis value at ref pixel
    'CRVAL2' #Second axis value at ref pixel
    'CTYPE1' #First axis coordinate type/units
    'CTYPE2' #Second axis coordinate type/units
    'CD1_1' #Partial of first axis coord wrt x
    'CD1_2' #Partial of first axis coord wrt y
    'CD2_1' #Partial of second axis coord wrt x
    'CD2_2' #Partial of second axis coord wrt y
    'LTV1' # X coord of first science pixel in SCI array
    'LTV2' # Y coord of first science pixel in SCI array
    'LTM1_1' # Ref pixel size in units of current pix size, X
    'LTM2_2' # Ref pix size in units of current pix size, Y
    '''
    NOTE: look through possible header keys for other instruments & compare
    '''

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

#------------------------------------------------------------------------------
# Various fns to fit to diffraction spike profiles
#------------------------------------------------------------------------------
def dso(x,A,cen,s1,s2,c,o):
    '''cen: center of diffraction pattern
    o: offset between diffraction center and double-slit peak
    s1: width of peak (cos)
    s2: width of envelope (sinc)
    c: additive offset
    A: normalization
    '''
    t1 = np.cos(s1*(x-cen)-o)**2.
    t2 = np.sinc(s2*(x-cen))**2.
    return A*t1*t2+c
    
def dsreg(x,A,cen,s1,s2,c):
    t1 = np.cos(s1*(x-cen))**2.
    t2 = np.sinc(s2*(x-cen))**2.
    return A*t1*t2+c
    
def gauss(x,A,mu,sig):
    '''
    gaussian with width sig, mean mu, normalization A and additive offset o
    '''
    expon = ((x-mu)**2.)/(2.*sig**2.)
    return A*np.exp(-expon)
    
def sinc2(x,a,p,s,o):
    '''
    np.sinc(x) = sin(pi*x)/(pi*x)
    this sinc2 fn uses 's' as a multiplicative width (small s = wider),
    'p' as a left-right offset, 'a' as a multiplicative prefactor.
    These values are then squared, and a vertical offset 'o' added.
    '''
    f=np.sinc(s*(x-p))
    return a*f**2+o
    
def threesinc2(x,a1,p1,s1,a2,p2,s2,a3,p3,s3,o):
    '''
    Additive function with 3 sinc^2 profiles for doubled spikes (ie WFPC2)
    Contains only a single offset term to prevent degeneracy
    '''
    return sinc2(x,a1,p1,s1,o/3.)+sinc2(x,a2,p2,s2,o/3.)+sinc2(x,a3,p3,s3,o/3.)

def twosinc2(x,a1,p1,s1,a2,p2,s2,o):
    '''
    Additive function with two sinc^2 profiles for doubled spikes (ie WFPC2)
    Contains only a single offset term to prevent degeneracy
    '''
    return sinc2(x,a1,p1,s1,o)+sinc2(x,a2,p2,s2,o)

#------------------------------------------------------------------------------
# Cheesy check for convergence of spike profile fits
'''
NOTE: NEEDS IMPROVEMENT
Perform a convergence check that is NOT cheesy.
'''
#------------------------------------------------------------------------------
def check_params(px,p0):
    '''
    px : parameters returned from curve_fit
    p0 : initial guess
    Raises an error if the fit amplitude is too different from the maximum of that image slice,
    or if the returned spike center is identically the guessed spike center (indicating the fit
    did not converge). Other conditions can be included here.
    '''
    if (px[0]<0.85*p0[0]) or (px[0]>1.15*p0[0]): raise ValueError
    #if px[2]<0.05: raise ValueError
    if px[1]==p0[1]: raise ValueError
    return

#------------------------------------------------------------------------------        
# Cheesy way to extract only the area around the spike max where values are decreasing
#------------------------------------------------------------------------------
def get_dec(vals):
    '''
    vals : 1D array-like of values. Function locates array maximum and identifies the area around
    the maximum where the array values are DECREASING. Returns the array slice, and the coords
    thereof in the original array.
    '''
    w = np.argmax(vals)
    wl = w+1
    wr = w-1
    desc = True
    end = len(vals)-1
    while desc:
        if wl<end-1:
            if vals[wl+1]<vals[wl]:
                if wl==end:
                    break
                else:
                    wl+=1
            else: desc = False
        else: break
    desc = True
    while desc:
        if wr>0:
            if vals[wr-1]<vals[wr]:
                if wr==0:
                    break
                else:
                    wr-=1
            else: desc = False
        else: break
    arr_inds = np.arange(wr,wl+1).astype('int')
    arr_slice = vals[wr:wl+1]
    return arr_inds,arr_slice

'''
NOTE: Function needs to remove bad indices more reliably and/or less arbitrarily
'''
def find_goods(rotated,horiz,vert,c_horiz,c_vert,bad_h,bad_v,cenguess,iterations=2):
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




'''
----------------------------------------------------------
NOTE:
NOTE: Main process loop; here only as an example of structure
NOTE:
----------------------------------------------------------
'''
split = False
#Actual process loop
if __name__=='__main__':
    for i,f in enumerate(fnames):
        print '\nOperating on file',f
        #if i>0: break
        img,drzangle = get_data(prefix+f,index=0,drz=False) #raw data
        square,diff = make_square(img)
        big,pad = padding(square) #padded; no rotation losses due to hitting edge of field
        rotated = derotate(big,drzangle) #rotated to have spikes vert & horiz
        #rotated = big
        cenguess = guess_center(rotated) #rough locations of said spikes
        print cenguess
        
        fitfns = [gauss]
        '''---NEEDS WORK---'''
        ints = np.zeros((3,2,6))
        for j,fitfn in enumerate(fitfns):
            print 'Fitting round ',j
            #Modify fit_both to accept a bool kwarg "split" for double spikes; default to False
            px,py,bx,by,cx,cy = fit_both(rotated,cenguess,fitfn)
            

            
            horiz = px[1,:]
            vert = py[1,:]
            inds = np.arange(rotated.shape[0])
            #print len(inds)
            #pdb.set_trace()
            #Will need two sets for each direction
            good_hs,good_horiz,good_vs,good_vert = find_goods(rotated,horiz,vert,cenguess,iterations=5)
            
            ##plt.imshow(np.log(rotated),origin='lower',cmap=plt.cm.gray)
            #plt.plot(good_hs,good_horiz,'bo')
            #plt.plot(good_vert,good_vs,'go')
            #plt.show()
            
            #will need two sets for each direction
            plh,plv,pph,ppv,pch,pcv = polynomial_fit(good_hs,good_horiz,good_vs,good_vert)
            ltup = tuple(plh)+tuple(plv)
            ptup = tuple(pph)+tuple(ppv)
            ctup = tuple(pch)+tuple(pcv)
            
            #More intersects possible; not all are useful
            ints[0,:,j] = find_ints(crossline,cenguess,ltup)
            ints[1,:,j] = find_ints(crossparab,cenguess,ptup)
            ints[2,:,j] = find_ints(crosscubic,cenguess,ctup)

            
            savename = prefix+'narrow_gauss/'+f
            snames.append(savename)
            angle=-drzangle+np.pi/4.
        
    
            np.savez_compressed(savename+str(j),img=rotated,orig=img,diff=diff,pad=pad,angle=angle,guess=cenguess,px=px,py=py,plh=plh,plv=plv,
                pph=pph,ppv=ppv,pch=pch,pcv=pcv,ints=ints,good_hs=good_hs,good_vs=good_vs,good_horiz=good_horiz,cx=cx,cy=cy,
                good_vert=good_vert,fname=f)
            #savename = prefix+obsnums[i]+'/'+obsids[i]+'/'+obsnums[i]+'_'+obsids[i]+'_'+lambdas[i]
        savename = prefix+'narrow_gauss/'+f
        snames.append(savename)
        angle=-drzangle+np.pi/4.
        inds = np.arange(rotated.shape[0])
        brightest = np.zeros((len(inds),2))
        lows = brightest.copy()
        highs = brightest.copy()
        cens = brightest.copy()
        for j in range(len(inds)):
            brightest[j,0]=np.argmax(rotated[:,j])
            brightest[j,1]=np.argmax(rotated[j,:])
            lows[j,0]=np.argmax(rotated[:brightest[j,0]-1,j])
            lows[j,1]=np.argmax(rotated[j,:brightest[j,1]-1])
            highs[j,0]=np.argmax(rotated[brightest[j,0]+2:,j])+brightest[j,0]+2
            highs[j,1]=np.argmax(rotated[j,brightest[j,1]+2:])+brightest[j,1]+2
            xs = [lows[j,0],brightest[j,0],highs[j,0]]
            ys = [lows[j,1],brightest[j,1],highs[j,1]]
            p,c = curve_fit(parab,xs,rotated[xs,j])
            cens[j,0]=-p[1]/2./p[0]
            p,c = curve_fit(parab,ys,rotated[j,ys])
            cens[j,1]=-p[1]/2./p[0]

        good_hs,good_horiz,good_vs,good_vert = find_goods(rotated,brightest[:,0],brightest[:,1],cenguess,iterations=2)
        
        plh,plv,pph,ppv,pch,pcv = polynomial_fit(good_hs,good_horiz,good_vs,good_vert)
        ltup = tuple(plh)+tuple(plv)
        ptup = tuple(pph)+tuple(ppv)
        ctup = tuple(pch)+tuple(pcv)
        
        ints[0,:,4] = find_ints(crossline,cenguess,ltup)
        ints[1,:,4] = find_ints(crossparab,cenguess,ptup)
        ints[2,:,4] = find_ints(crosscubic,cenguess,ctup)
        
        good_hs,good_horiz,good_vs,good_vert = find_goods(rotated,cens[:,0],cens[:,1],cenguess,iterations=2)
        
        plh,plv,pph,ppv,pch,pcv = polynomial_fit(good_hs,good_horiz,good_vs,good_vert)
        ltup = tuple(plh)+tuple(plv)
        ptup = tuple(pph)+tuple(ppv)
        ctup = tuple(pch)+tuple(pcv)
        
        ints[0,:,5] = find_ints(crossline,cenguess,ltup)
        ints[1,:,5] = find_ints(crossparab,cenguess,ptup)
        ints[2,:,5] = find_ints(crosscubic,cenguess,ctup)
        #pdb.set_trace()
        np.savez_compressed(savename,img=rotated,orig=img,diff=diff,pad=pad,angle=angle,guess=cenguess,px=px,py=py,plh=plh,plv=plv,
            pph=pph,ppv=ppv,pch=pch,pcv=pcv,ints=ints,good_hs=good_hs,good_vs=good_vs,good_horiz=good_horiz,
            good_vert=good_vert,fname=f,brightest=brightest,lows=lows,highs=highs)
       
        #Fix the plotting for split spikes
        plt.plot(inds,horiz,'bo',vert,inds,'go')
        plt.plot(inds,line(inds,*plh),'b-',line(inds,*plv),inds,'g-')
        plt.plot(inds,parab(inds,*pph),'b--',parab(inds,*ppv),inds,'g--')
        plt.plot(inds,cubic(inds,*pch),'b:',cubic(inds,*pcv),inds,'g:')
        plt.xlim([min(inds),max(inds)])
        plt.ylim([min(inds),max(inds)])
        plt.savefig(savename+'.svg')
        plt.clf()
        plt.plot(1174,1174,'bo')
        plt.plot(ints[0,0,0],ints[0,1,0],'ro',label='Gauss')
        #plt.plot(ints[0,0,1],ints[0,1,1],'yo')
        #plt.plot(ints[0,0,2],ints[0,1,2],'go',label='Gaussian')
        #plt.plot(ints[0,0,3],ints[0,1,3],'co',label='sinc^2')
        plt.plot(ints[0,0,4],ints[0,1,4],'mo',label='brightest pixel trace')
        plt.plot(ints[0,0,5],ints[0,1,5],'ko',label='Bond parabola')
        #pdb.set_trace()
        #plt.interactive(True)
        plt.legend()
        #plt.show()
        #pdb.set_trace()
        plt.savefig(savename+'ints.svg')
        plt.clf()
        print 'Saved file: '+savename+'.npz'
        print ints[:,:,0]

        
        
        
        
        
        
        
        
        
        
        
        #    print 'Something went wrong for',f
        #    errors.append(f)
    np.savez_compressed(prefix+'narrow_gauss/info',snames=snames,errors=errors)
    #To do: Test this, then apply it to HPF, BPF, and smoothed images

