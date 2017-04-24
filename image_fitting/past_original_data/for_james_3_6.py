import numpy as np
import matplotlib.pyplot as plt

from spikefit_gen3 import *

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

with np.load('F:/PSF_modeling/acswfc1/acs_wfc100_psf.fits0.npz') as d:
    #px = d['px']
    pyo = d['py'][:,1600]
    img = d['img']
    cenguess = d['guess']
    #good_hs = d['good_hs']
    #good_vs = d['good_vs']
    #good_horiz = d['good_horiz']
    #good_vert = d['good_vert']
    
    
inds = np.arange(img.shape[0])

bounds = (np.array([0.,0.,0.,0.,-np.inf,-np.pi/2.]),np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.pi/2.]))

fitcoords = np.arange(int(cenguess[0])-40,int(cenguess[0])+40)


#with open('vert_80px.txt','w') as f:
#    for c in fitcoords:
#        f.write(str(c)+'  '+str(img[1600,c])+'  '+str(dso(c,*pyo))+'\n')

cenguess = guess_center(img)
xguess = cenguess[0]
yguess = cenguess[1]

fitcoords = np.arange(1100,1251)
fitvals = img[fitcoords,1600]

p0=(np.max(img[:,1600]),yguess,.6,.1,0.,.2)
px,cx = curve_fit(dso,fitcoords,fitvals,p0=p0,bounds=bounds,max_nfev=1000*len(fitcoords))

fitvals = img[1600,fitcoords]
p0=(np.max(img[1600,:]),xguess,.6,.1,0.,.2)

#curve_fit(fitfn,fitcoords,fitvals,p0=p0,bounds=bounds,max_nfev=1000*len(fitcoords))

py,cy = curve_fit(dso,fitcoords,fitvals,p0=p0,bounds=bounds,max_nfev=1000*len(fitcoords))

plt.interactive(True)
plt.plot(img[:,1600])
#pdb.set_trace()
plt.plot(img[1600,:])
#pdb.set_trace()
plt.plot(dso(inds,*px))
#pdb.set_trace()
plt.plot(dso(inds,*py))
pdb.set_trace()


        
#with open('vert_150px.txt','w') as f:
#    for c in fitcoords:
#        f.write(str(c)+'  '+str(img[1600,c])+'  '+str(dso(c,*py))+'\n')
        
#with open('horiz_150px.txt','w') as f:
#    for c in fitcoords:
#        f.write(str(c)+'  '+str(img[c,1600])+'  '+str(dso(c,*px))+'\n')
        
#with open('params.txt','w') as f:
#    f.write(str(pyo[0])+'  '+str(pyo[1])+'  '+str(pyo[2])+'  '+str(pyo[3])+'  '+str(pyo[4])+'  '+str(pyo[5])+'\n')
#    f.write(str(py[0])+'  '+str(py[1])+'  '+str(py[2])+'  '+str(py[3])+'  '+str(py[4])+'  '+str(py[5])+'\n')
#    f.write(str(px[0])+'  '+str(px[1])+'  '+str(px[2])+'  '+str(px[3])+'  '+str(px[4])+'  '+str(px[5])+'\n')