import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,brentq,root

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