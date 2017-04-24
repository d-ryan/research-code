import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,brentq,root


#For fixing x,y
#if j<2: horizontal
#else: vertical


fnames = ['j70s','j71s','j72s','j73s']
inds=np.arange(0.,2462.,1.)
guess=np.array([2400,2400])


#Fit and intersect functions
#Sub: for intersections of near-parallel lines
#Cross: for intersections of near-perpendicular curves
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
    
    

with np.load('spikedata.npz') as d:
    raw=d['raw']
    trim=d['trim']
    
for i in range(4): # for each image
    #Generate some arrays
    #Fits
    pl=np.zeros((2,4))
    plh=np.zeros((2,8))
    cl=np.zeros((2,2,4))
    clh=np.zeros((2,2,8))
    pp=np.zeros((3,4))
    cp=np.zeros((3,3,4))
    pc=np.zeros((4,4))
    cc=np.zeros((4,4,4))
    
    #intersections
    fullintsl=np.zeros((6,2))
    fullintsp=np.zeros((6,2))
    fullintsc=np.zeros((6,2))
    halfints=np.zeros((28,2))
    
    #fill in fits
    for j in range(4):
        spike=trim[i,j,0,:]
        
        left=spike[:1200]
        leftinds=inds[:1200][left>0]
        left=left[left>0]
        right=spike[1200:]
        rightinds=inds[1200:][right>0]
        right=right[right>0]
        
        useinds=inds[spike>0]
        spike=spike[spike>0]

        pl[:,j],cl[:,:,j]=curve_fit(line,useinds,spike)
        pp[:,j],cp[:,:,j]=curve_fit(parab,useinds,spike)
        pc[:,j],cc[:,:,j]=curve_fit(cubic,useinds,spike)
        
        plh[:,2*j],clh[:,:,2*j]=curve_fit(line,leftinds,left)
        plh[:,2*j+1],clh[:,:,2*j+1]=curve_fit(line,rightinds,right)

    
    #Full intersections
    z=0
    for j in range(3):
        for k in range(j+1,4):
            if (j<2 and k<2):
                subtupl=tuple(pl[:,j])+tuple(pl[:,k])
                try:
                    x=brentq(subline,1000,1400,args=subtupl)
                    fullintsl[z,0]=x
                    fullintsl[z,1]=line(x,*pl[:,j])
                except ValueError:
                    fullintsl[z,0]=-1
                    fullintsl[z,1]=-1
                
                subtupp=tuple(pp[:,j])+tuple(pp[:,k])
                try:
                    x=brentq(subparab,1000,1400,args=subtupp)
                    fullintsp[z,0]=x
                    fullintsp[z,1]=parab(x,*pp[:,j])
                except ValueError:
                    fullintsp[z,0]=-1
                    fullintsp[z,1]=-1
                
                subtupc=tuple(pc[:,j])+tuple(pc[:,k])
                try:
                    x=brentq(subcubic,1000,1400,args=subtupc)
                    fullintsc[z,0]=x
                    fullintsc[z,1]=cubic(x,*pc[:,j])
                except ValueError:
                    fullintsc[z,0]=-1
                    fullintsc[z,1]=-1
            elif (j>=2 and k>=2):
                subtupl=tuple(pl[:,j])+tuple(pl[:,k])
                try:
                    x=brentq(subline,1000,1400,args=subtupl)
                    fullintsl[z,0]=line(x,*pl[:,j])
                    fullintsl[z,1]=x
                except ValueError:
                    fullintsl[z,0]=-1
                    fullintsl[z,1]=-1
                
                subtupp=tuple(pp[:,j])+tuple(pp[:,k])
                try:
                    x=brentq(subparab,1000,1400,args=subtupp)
                    fullintsp[z,0]=parab(x,*pp[:,j])
                    fullintsp[z,1]=x
                except ValueError:
                    fullintsp[z,0]=-1
                    fullintsp[z,1]=-1
                
                subtupc=tuple(pc[:,j])+tuple(pc[:,k])
                try:
                    x=brentq(subcubic,1000,1400,args=subtupc)
                    fullintsc[z,0]=cubic(x,*pc[:,j])
                    fullintsc[z,1]=x
                except ValueError:
                    fullintsc[z,0]=-1
                    fullintsc[z,1]=-1
            else:
                crosstupl=tuple(pl[:,j])+tuple(pl[:,k])
                soln=root(crossline,guess,args=crosstupl)
                if soln.success:
                    fullintsl[z,0]=soln.x[0]
                    fullintsl[z,1]=soln.x[1]
                else:
                    fullintsl[z,0]=-1
                    fullintsl[z,0]=-2
                crosstupp=tuple(pp[:,j])+tuple(pp[:,k])
                soln=root(crossparab,guess,args=crosstupp)
                if soln.success:
                    fullintsp[z,0]=soln.x[0]
                    fullintsp[z,1]=soln.x[1]
                else:
                    fullintsp[z,0]=-1
                    fullintsp[z,1]=-1
                crosstupc=tuple(pc[:,j])+tuple(pc[:,k])
                soln=root(crosscubic,guess,args=crosstupc)
                if soln.success:
                    fullintsc[z,0]=soln.x[0]
                    fullintsc[z,1]=soln.x[1]
                else:
                    fullintsc[z,0]=-1
                    fullintsc[z,1]=-1
            
            z+=1


    #Half intersections
    z=0
    for j in range(7):
        for k in range(j+1,8):
            if (j<4 and k<4):
                subtupl=tuple(plh[:,j])+tuple(plh[:,k])
                try:
                    x=brentq(subline,1000,1400,args=subtupl)
                    halfints[z,0]=x
                    halfints[z,1]=line(x,*plh[:,j])
                except ValueError:
                    halfints[z,0]=-1
                    halfints[z,1]=-1
            elif (j>=4 and k>=4):
                subtupl=tuple(plh[:,j])+tuple(plh[:,k])
                try:
                    x=brentq(subline,600,1800,args=subtupl)
                    halfints[z,0]=line(x,*plh[:,j])
                    halfints[z,1]=x
                except ValueError:
                    halfints[z,0]=-1
                    halfints[z,1]=-1
            else:
                crosstupl=tuple(plh[:,j])+tuple(plh[:,k])
                soln=root(crossline,guess,args=crosstupl)
                if soln.success:
                    halfints[z,0]=soln.x[0]
                    halfints[z,1]=soln.x[1]
                else:
                    halfints[z,0]=-1
                    halfints[z,1]=-1
            
            z+=1
    for j in range(28):
        if halfints[j,0]==-1 or halfints[j,0]==0: print i,j,' half'
        if j<6:
            if fullintsl[j,0]==-1 or fullintsl[j,0]==0: print i,j,' lin'
            if fullintsp[j,0]==-1 or fullintsp[j,0]==0: print i,j,' quad'
            if fullintsc[j,0]==-1 or fullintsc[j,0]==0: print i,j,' cubic'
    
    np.savez_compressed(fnames[i]+'intdata',half=halfints,lin=fullintsl,par=fullintsp,cub=fullintsc,pl=pl,pp=pp,pc=pc,plh=plh)