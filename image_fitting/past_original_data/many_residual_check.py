import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from spikefit_gen_dev import get_data,line,parab,cubic,crossline,crossparab,crosscubic,polynomial_fit,find_ints

def resid(x,y,func,params):
    residuals = y-func(x,*params)

if __name__=='__main__':
    #prefix = 'F:/sirius_astrometry/'
    prefix = 'F:/hr8799/'

    with np.load(prefix+'info.npz') as d:
        snames=list(d['snames'])

    all_ints = np.zeros((len(snames),3,2))
    fnames = []
    angles = []
    pads = []
    bigshapes = []
    shapes = []
    diffs = []
    #pxs = []
    #pys = []
    #plhs = []
    #plvs = []
    #pphs = []
    #ppvs = []
    #pchs = []
    #pcvs = []
    
    for i,s in enumerate(snames):
        with np.load(s+'.npz') as d:
            ints=d['ints']
            fnames.append(str(d['fname']))
            angles.append(d['angle'])
            pads.append(d['pad'])
            bigshapes.append(len(d['px'][1,:]))
            diffs.append(d['diff'])
            #pxs.append(d['px'])
            #pys.append(d['py'])
            #plhs.append(d['plh'])
            #plvs.append(d['plv'])
            #pphs.append(d['pph'])
            #ppvs.append(d['ppv'])
            #pchs.append(d['pch'])
            #pcvs.append(d['pcv'])
            shp = len(d['px'][1,:])
            px = d['px']
            py = d['py']
            plh = d['plh']
            plv = d['plv']
            pph = d['pph']
            ppv = d['ppv']
            pch = d['pch']
            pcv = d['pcv']
            good_hs = d['good_hs']
            good_vs = d['good_vs']
            good_horiz = d['good_horiz']
            good_vert = d['good_vert']
        
        
        
        inds = np.arange(shp)
        #for i in inds: print cubic(i,*pch)
        #print pch
        fig = plt.figure()
        fig.set_size_inches(18,8,forward=True)
        ax = fig.add_subplot(221)
        #plt.plot(inds,px[1,:],'b-')
        plt.plot(good_hs,good_horiz,'bo')
        plt.plot(good_hs,line(good_hs,*plh),'b--',label='Linear')
        plt.plot(good_hs,parab(good_hs,*pph),'b:',label='Quadratic')
        plt.plot(good_hs,cubic(good_hs,*pch),'b-.',label='Cubic')
        if plh[0]<=0:
            leg = ax.legend(loc=1)
        else:
            leg = ax.legend(loc=4)
        ax.add_artist(leg)
        plt.title('Horizontal Spike')
        plt.ylabel('y pos. [pix] of spike')
        ax2 = fig.add_subplot(222)
        #plt.plot(inds,px[1,:],'g-')
        plt.plot(good_vs,good_vert,'go')
        plt.plot(good_vs,line(good_vs,*plv),'g--',label='Linear')
        plt.plot(good_vs,parab(good_vs,*ppv),'g:',label='Quadratic')
        plt.plot(good_vs,cubic(good_vs,*pcv),'g-.',label='Cubic')
        if plv[0]<=0:
            leg2 = ax2.legend(loc=1)
            print 'negative slope'
        else:
            leg2 = ax2.legend(loc=4)
            print 'positive slope'
        ax2.add_artist(leg2)
        plt.title('Vertical Spike')
        plt.ylabel('x pos. [pix] of spike')
        ax3 = fig.add_subplot(223)
        plt.axhline(1,color='k')
        plt.axhline(0,color='k')
        plt.axhline(-1,color='k')
        plt.plot(good_hs,good_horiz-line(good_hs,*plh)+1,'b--',label='Linear')
        plt.plot(good_hs,good_horiz-parab(good_hs,*pph),'b:',label='Quadratic')
        plt.plot(good_hs,good_horiz-cubic(good_hs,*pch)-1,'b-.',label='Cubic')
        #plt.legend()
        plt.xlabel('x [pix]')
        plt.ylabel('Residuals')
        ax4 = fig.add_subplot(224)
        plt.axhline(1,color='k')
        plt.axhline(0,color='k')
        plt.axhline(-1,color='k')
        plt.plot(good_vs,good_vert-line(good_vs,*plv)+1,'g--',label='Linear')
        plt.plot(good_vs,good_vert-parab(good_vs,*ppv),'g:',label='Quadratic')
        plt.plot(good_vs,good_vert-cubic(good_vs,*pcv)-1,'g-.',label='Cubic')
        #plt.legend()
        plt.xlabel('y [pix]')
        plt.ylabel('Residuals')
        plt.savefig(s+'resid.svg')
        #plt.show()
        plt.close()