import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from spikefit_gen_dev import get_data
import pdb




pre_vals = ['ibk7','ibti','ic1k','ica1']
end_vals = ['03010','03020','03030','03040']

guesses = [(600.,658.),(644.,622.),(650.,623.),(373.,453.)]


def gauss2d(x,y,A,xc,yc,sig,o):
    num = (x-xc)**2. + (y-yc)**2.
    denom = 2*sig**2.
    return A*np.exp(-num/denom)+o
    

def cen_guess(params,img):
    assert isinstance(img,np.ndarray)
    xc = params[0]
    yc = params[1]
    A = params[2]
    sig = params[3]
    o = params[4]
    xn,yn = np.round([xc,yc])
    xmin = xn-10.
    xmax = xn+10.
    ymin = yn-10.
    ymax = yn+10.
    xs = np.arange(xmin,xmax+1.)
    ys = np.arange(ymin,ymax+1.)
    sq_diffs = 0.
    for x in xs:
        for y in ys:
            image_val = img[y,x]
            calc_val = gauss2d(x,y,A,xc,yc,sig,o)
            sq_diffs += (image_val-calc_val)**2.
    return sq_diffs
    
    
#initial guess

A = 1800.
sig = 1.
o = 1.5
results = []
fnames = []
for i,pre in enumerate(pre_vals):
    #if i>0: continue
    for j,end in enumerate(end_vals):
        #if j>0: continue
        img,drz = get_data('F:/sirius_astrometry/'+pre+end+'_drz.fits',index=1,drz=False)
        xc = guesses[i][0]
        yc = guesses[i][1]
        x0 = np.array([xc,yc,A,sig,o])
        bounds = ((xc-10,xc+10),(yc-10,yc+10),(None,None),(None,None),(None,None))
        result = minimize(cen_guess,x0,args=(img,),bounds=bounds)
        results.append(result.x)
        fnames.append(pre+end+'_drz.fits')
        
        
np.savez_compressed('sir_B_cens',results=results,fnames=fnames)

pdb.set_trace()
        