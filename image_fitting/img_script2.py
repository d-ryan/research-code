import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
    #with np.load('rot'+fnames[i]+'.npz') as d:
    #    img=d['pos']
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
        
        pl,cl=curve_fit(line,useinds,spike)
        pp,cp=curve_fit(parab,useinds,spike)
        pc,cc=curve_fit(cubic,useinds,spike)
        
        
        
        linres=spike-line(useinds,*pl)
        parres=spike-parab(useinds,*pp)
        cubres=spike-cubic(useinds,*pc)
        
        
        
        #Plotting
        
        fig=plt.figure()
        ax=fig.add_subplot(231)
        plt.plot(leftinds,left,'b-',rightinds,right,'b-')
        plt.plot(inds,line(inds,*pl),'r-')
        plt.title('Full Linear Fit, spike '+str(j+1))
        ax=fig.add_subplot(234)
        plt.plot(useinds,linres,'b-')
        plt.ylabel('Residuals')
        ax=fig.add_subplot(232)
        plt.plot(leftinds,left,'b-',rightinds,right,'b-')
        plt.plot(inds,parab(inds,*pp),'r-')
        plt.title('Full Quadratic Fit, spike '+str(j+1))
        ax=fig.add_subplot(235)
        plt.plot(useinds,parres,'b-')
        ax=fig.add_subplot(233)
        plt.plot(leftinds,left,'b-',rightinds,right,'b-')
        plt.plot(inds,cubic(inds,*pc),'r-')
        plt.title('Full Cubic Fit, spike '+str(j+1))
        ax=fig.add_subplot(236)
        plt.plot(useinds,cubres,'b-')
        savestr1='j'+str(70+i)+'s/spike'+str(j+1)+'_full.pdf'
        savestr2='j'+str(70+i)+'s/spike'+str(j+1)+'_full.svg'
        plt.savefig(savestr1)
        plt.savefig(savestr2)
        #plt.show()