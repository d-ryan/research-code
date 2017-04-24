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

prefix = 'F:/BetaPic/wfc3/'

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
    if filename[-11:-9]=='wf':
        elems=filename.split('_')
        obsnum=int(elems[1])
        fnames.append(filename)
        obsnums.append(elems[1])
        obsids.append(elems[2])
        lambdas.append(elems[4])

nfiles = len(fnames)        
print 'Operating on '+str(nfiles)+' files.'

#Make directories if needed
for i,id in enumerate(obsids):
    try:
        os.chdir(prefix+obsnums[i]+'/'+id)
        os.chdir(prefix)
    except WindowsError:
        os.mkdir(prefix+obsnums[i]+'/'+id)
        print 'Made new directory: '+obsnums[i]+'/'+id

print 'All needed directories present.'
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
    shp = img.shape[0]
    bigshp = np.ceil(shp*np.sqrt(2.))
    pad = (bigshp-shp)//2
    big = np.zeros((bigshp,bigshp))
    big[pad:pad+shp,pad:pad+shp]=img
    return big,pad
    
def derotate(img,drzangle):
    #drzangle is recovered from headers
    #and gives amt by which the original, diagonal-spike img
    #was rotated to give a north-up image
    #de-rotate by this amount, then rotate pi/4 to give vert & horiz spikes
    derot_angle = -drzangle+np.pi/4.
    cen = img.shape[0]//2
    rot = _warp_fast(img,rotator(derot_angle,cen),order=2)
    return rot
    
def guess_center(img):
    img = np.nan_to_num(img.copy())
    xsum = np.sum(img,axis=0) #collapse over 0th axis (i.e. stack rows, preserve columns)
    pixels = np.arange(float(img.shape[0]))
    ysum = np.sum(img,axis=1) #collapse over 1st axis (stack columns, preserve rows)
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
    shift0=np.array([[1,0,-center],[0,1,-center],[0,0,1]])
    shift1=np.array([[1,0,center],[0,1,center],[0,0,1]])
    R=np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
    return shift1.dot(R).dot(shift0)

def get_data(filename):
    hdulist=fits.open(filename)
    rawdata=hdulist[1].data
    drzangle = np.radians(hdulist[0].header['D001ROT'])
    hdulist.close()
    return rawdata,drzangle
    
def gauss(x,mu,sig,A):
    norm = A/np.sqrt(2.*np.pi)/sig
    expon = ((x-mu)**2.)/(2.*sig**2.)
    return norm*np.exp(-expon)
    
def sinc2(x,a,p,s,o):
    f=a*np.sinc(s*(x-p))
    return f**2+o
    

    
    
    
    
    
    
#------------------------------------------------------------------------------        
# Function that does the row-by-row & column-by-column fitting
#------------------------------------------------------------------------------
def fit_both(img,cenguess):
    xguess = cenguess[0]
    yguess = cenguess[1]
    shp = img.shape[0]
    cen=shp//2
    inds = np.arange(shp)
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
                
    '''            
    #After running the loop
    #Check for spike widths (1/s==1/px[_,2]) more than 3 sigma wider than they should be
    ave_width = np.mean(1./px[:,2])
    std_width = np.std(1./px[:,2])
    for i,s in enumerate(px[:,2]):
        if np.abs(1./s - ave_width) > 3*std_width:
            if i not in badsx: badsx.append(i)
    '''
    #fit vertical spike
    print 'Fitting vertical'
    for i in inds: #loop over columns
        nz = np.where(img[i,:]>0)[0]
        imod = i-cen
        offset = img[i,nz]-np.mean(img[i,nz])
        
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
                
    '''
    #After running the loop
    #Check for spike widths (1/s==1/px[_,2]) more than 3 sigma wider than they should be
    ave_width = np.mean(1./py[:,2])
    std_width = np.std(1./py[:,2])
    for i,s in enumerate(py[:,2]):
        if np.abs(1./s - ave_width) > 3*std_width:
            if i not in badsy: badsy.append(i)
            
    badsx = sorted(badsx[:])
    badsy = sorted(badsy[:])
    '''
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
    medh = np.median(fith)
    medv = np.median(fitv)
    
    sigh = np.abs(fith-medh)+.1
    sigv = np.abs(fitv-medv)+.1
    #plt.plot(i_h,fith,'bo',fitv,i_v,'go')
    #plt.xlim([min(inds),max(inds)])
    #plt.ylim([min(inds),max(inds)])
    #plt.show()
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


def find_goods(rotated,horiz,vert):

    bad_hs = []
    bad_vs = []
    
    shp = rotated.shape
    inds = np.arange(shp[0])
    
    for ind in inds:
        o_h = np.round(horiz[ind])
        o_v = np.round(vert[ind])
        if o_h>=shp[0]: bad_hs.append(ind)
        elif o_h<0: bad_hs.append(ind)
        elif rotated[o_h,ind]<=0: bad_hs.append(ind)
        
        if o_v>=shp[1]: bad_vs.append(ind)
        elif o_v<0: bad_vs.append(ind)
        elif rotated[ind,o_v]<=0: bad_vs.append(ind)
    
    temp = list(inds.copy())
    for ind in temp[:]:
        if ind in bad_hs: temp.remove(ind)
    good_hs = temp[:]
    temp = list(inds.copy())
    for ind in temp[:]:
        if ind in bad_vs: temp.remove(ind)
    good_vs = temp[:]
    
    good_horiz = horiz[good_hs]
    good_vert = vert[good_vs]
    
    hstd = np.std(good_horiz)
    hmean = np.mean(good_horiz)
    vstd = np.std(good_vert)
    vmean = np.mean(good_vert)
    
    where_good_hs = np.where(np.abs(good_horiz-hmean)<1.5*hstd)[0]
    where_good_vs = np.where(np.abs(good_vert-vmean)<1.5*vstd)[0]
    
    good_hs = np.array(good_hs)[where_good_hs]
    good_vs = np.array(good_vs)[where_good_vs]
    
    good_horiz = good_horiz[where_good_hs]
    good_vert = good_vert[where_good_vs]
    
    return good_hs,good_horiz,good_vs,good_vert
    

#Actual process loop
if __name__=='__main__':
    for i,f in enumerate(fnames):
        print '\nOperating on file',f
        #if i<7: continue
        #if i>8: break
        img,drzangle = get_data(prefix+f) #raw data
        big,pad = padding(img) #padded; no rotation losses due to hitting edge of field
        rotated = derotate(big,drzangle) #rotated to have spikes vert & horiz
        
        cenguess = guess_center(rotated) #rough locations of said spikes
        print cenguess
        px,py,bx,by = fit_both(rotated,cenguess)
        horiz = px[1,:]
        vert = py[1,:]
        inds = np.arange(len(horiz))
        
        good_hs,good_horiz,good_vs,good_vert = find_goods(rotated,horiz,vert)
        
        plh,plv,pph,ppv,pch,pcv = polynomial_fit(good_hs,good_horiz,good_vs,good_vert)
        
        ltup = tuple(plh)+tuple(plv)
        ptup = tuple(pph)+tuple(ppv)
        ctup = tuple(pch)+tuple(pcv)
        
        ints = np.zeros((3,2))
        
        ints[0,:] = find_ints(crossline,cenguess,ltup)
        ints[1,:] = find_ints(crossparab,cenguess,ptup)
        ints[2,:] = find_ints(crosscubic,cenguess,ctup)
        
        savename = prefix+obsnums[i]+'/'+obsids[i]+'/'+obsnums[i]+'_'+obsids[i]+'_'+lambdas[i]
        snames.append(savename)
        angle=-drzangle+np.pi/4.
        np.savez_compressed(savename,img=rotated,pad=pad,angle=angle,guess=cenguess,px=px,py=py,plh=plh,plv=plv,
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
    np.savez_compressed('info_bpf',snames=snames,errors=errors)
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