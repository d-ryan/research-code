import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

location='F:/BetaPic/acs/raw/'
prefix='hst_9861_'
numlist=['01','04','05','06','07','08','09','10']
suffix='_acs_wfc_f'
wvlist=['435','606','814']
suffix2='w_drz.fits'

pairs=[(0,1),(0,2),(1,2),(0,),(2,),(1,),(0,2),(1,2)]

side=['LEFT','RIGHT','TOP','BOT']


def line(x,m,b):
    return m*x+b

def parab(x,a,b,c):
    return a*x**2 + b*x + c

def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d


if __name__=='__main__':
    for i,n in enumerate(numlist): #Image number
        for j in pairs[i]: #Wavelength
            #if (i!=6 and j!=0): continue
            fname = location+n+'/spike_master_'+n+'_'+wvlist[j]+'.npz'
            with np.load(fname) as d:
                raw=d['raw']
                trim=d['trim']
                
            xh=np.where(trim[0,0,:]>0)[0]
            xv=np.where(trim[1,0,:]>0)[0]
            yh=trim[0,0,xh]
            yv=trim[1,0,xv]
            horiz=np.zeros((2,len(xh)))
            vert=np.zeros((2,len(xv)))
            horiz[0,:]=xh.copy()
            horiz[1,:]=yh.copy()
            vert[0,:]=xv.copy()
            vert[1,:]=yv.copy()
            plh,clh=curve_fit(line,xh,yh)
            pph,cph=curve_fit(parab,xh,yh)
            pch,cch=curve_fit(cubic,xh,yh)
            
            plv,clv=curve_fit(line,xv,yv)
            ppv,cpv=curve_fit(parab,xv,yv)
            pcv,ccv=curve_fit(cubic,xv,yv)
            
            ylh=line(xh,*plh)
            yph=parab(xh,*pph)
            ych=cubic(xh,*pch)
            
            ylv=line(xv,*plv)
            ypv=parab(xv,*ppv)
            ycv=cubic(xv,*pcv)
            
            reslh=ylh-yh
            resph=yph-yh
            resch=ych-yh
            
            reslv=ylv-yv
            respv=ypv-yv
            rescv=ycv-yv
            
            fig=plt.figure()
            ax1=fig.add_subplot(221)
            plt.plot(xh,yh,'k-')
            plt.plot(xh,ylh,'b-')
            plt.plot(xh,yph,'g-')
            plt.plot(xh,ych,'r-')
            plt.title('Horizontal Spike')
            
            ax2=fig.add_subplot(222)
            plt.plot(xv,yv,'k-')
            plt.plot(xv,ylv,'b-')
            plt.plot(xv,ypv,'g-')
            plt.plot(xv,ycv,'r-')
            plt.title('Vertical Spike [rotated 90 degrees]')
            
            ax3=fig.add_subplot(223)
            plt.plot(xh,resph,'b-')
            plt.plot(xh,resch+1,'g-')
            #plt.plot(xh,resch+2,'r-')
            plt.plot(xh,resch-resph+2,'r-')
            plt.xlabel('pixels [horizontal]')
            
            ax4=fig.add_subplot(224)
            plt.plot(xv,respv,'b-')
            plt.plot(xv,rescv+1,'g-')
            #plt.plot(xv,rescv+2,'r-')
            plt.plot(xv,rescv-respv+2,'r-')
            plt.xlabel('pixels [vertical]')
            
            plt.savefig(location+n+'/residuals_'+n+'_'+wvlist[j]+'.pdf')
            plt.savefig(location+n+'/residuals_'+n+'_'+wvlist[j]+'.svg')
            
            np.savez_compressed(location+n+'/fits_'+n+'_'+wvlist[j],horiz=horiz,vert=vert,
                plh=plh,pph=pph,pch=pch,plv=plv,ppv=ppv,pcv=pcv)
            
            #plt.show()
            print n,wvlist[j]