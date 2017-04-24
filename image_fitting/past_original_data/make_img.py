import numpy as np
import numpy as np
from astropy.io import fits
from skimage.transform._warps_cy import _warp_fast
from scipy.optimize import curve_fit,root
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
from spikefit_wfpc_dev import get_data,padding,derotate,line,parab,cubic




#npz_file = 'F:/BetaPic/wfpc2/05202/01/05202_01_f814w.npz'

def make_img(npz_file):

    prefix = 'F:/BetaPic/wfpc2/'

    with np.load(npz_file+'.npz') as d:
        rotated=d['img']
        px=d['px']
        py=d['py']
        plh=d['plh']
        plv=d['plv']
        pph=d['pph']
        ppv=d['ppv']
        pch=d['pch']
        pcv=d['pcv']
        good_hs=d['good_hs']
        good_vs=d['good_vs']
        good_horiz=d['good_horiz']
        good_vert=d['good_vert']
        fname = str(d['fname'])

    #pdb.set_trace()
    #img,drzangle = get_data(prefix+fname)
    #big,pad = padding(img)
    #rotated = derotate(big,drzangle)

    inds=np.arange(rotated.shape[0])

    #print len(inds)
    #print len(px[1,:])

    plt.imshow(np.log(rotated),origin='lower')
    plt.plot(good_hs,good_horiz,'bo')
    plt.plot(good_vert,good_vs,'go')
    plt.plot(inds,line(inds,*plh),'b-')
    plt.plot(inds,parab(inds,*pph),'b--')
    plt.plot(inds,cubic(inds,*pch),'b:')
    plt.plot(line(inds,*plv),inds,'g-')
    plt.plot(parab(inds,*ppv),inds,'g--')
    plt.plot(cubic(inds,*pcv),inds,'g:')
    plt.show()