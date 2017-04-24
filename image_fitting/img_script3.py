import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''FOR TUESDAY:
The "top" fits are INCORRECT. Re-do.

Then re-do datascript and img_script(s).

'''


''' relevant stuff:
Residuals of 1st, 2nd, 3rd order one-sided and two-sided fits
8*4*3 + 4*4*3 = 144 relevant sequences of residuals

Plot 2x2: Data+fit, linres,parres,cubres (2*4*4) = 32

save: fitdata.npz
contains: individual spikes & fits & residuals
          combined spikes & fits & residuals


Plot images, with spike fits overlaid (4)

Plot images w/ one-sided spikefit fits (4)
plot images w/ two-sided spikefit fits (4)




'''



fnames = ['j70s','j71s','j72s','j73s']
inds=np.arange(0.,2462.,1.)

def line(x,m,b):
    return m*x+b

def parab(x,a,b,c):
    return a*x**2 + b*x + c

def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

with np.load('spikedata.npz') as d:
    raw=d['raw']
    trim=d['trim']
    lin=d['lin']
    par=d['par']
    cub=d['cub']



#Four images
for i in range(4):
    with np.load('rot'+fnames[i]+'.fits.npz') as d:
        img=d['pos']
    fig=plt.figure()
    ax1=fig.add_subplot(121)
    ax1.imshow(np.log(img),origin='lower')
    ax2=fig.add_subplot(122)
    ax2.imshow(np.log(img),origin='lower')
    for j in range(4):
        
        #Calculation
        
        spike=trim[i,j,0,:]

        left=spike[:1200]
        leftinds=inds[:1200][left>0]
        left=left[left>0]
        right=spike[1200:]
        rightinds=inds[1200:][right>0]
        right=right[right>0]

        
        useinds=inds[spike>0]
        spike=spike[spike>0]
        
        pc,cc=curve_fit(cubic,useinds,spike)
        
        
        
        #Plotting
        
        if j==0:
            ax2.plot(leftinds,left,'g-',rightinds,right,'g-')
            ax2.plot(inds,cubic(inds,*pc),'g--')
        elif j==1:
            ax2.plot(leftinds,left,'b-',rightinds,right,'b-')
            ax2.plot(inds,cubic(inds,*pc),'b--')
        elif j==2:
            ax2.plot(left,leftinds,'c-',right,rightinds,'c-')
            ax2.plot(cubic(inds,*pc),inds,'c--')
        else:
            ax2.plot(left,leftinds,'r-',right,rightinds,'r-')
            ax2.plot(cubic(inds,*pc),inds,'r--')
        
        
    
    savestr='j'+str(70+i)+'s/fitted_img.pdf'
    plt.savefig(savestr)
    savestr='j'+str(70+i)+'s/fitted_img.svg'
    plt.savefig(savestr)
    plt.show()