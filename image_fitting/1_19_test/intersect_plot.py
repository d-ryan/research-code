import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



fnames = ['j70s','j71s','j72s','j73s']
inds=np.arange(0.,2462.,1.)

def line(x,m,b):
    return m*x+b

def parab(x,a,b,c):
    return a*x**2 + b*x + c

def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

#Plot Color-coding: Blue-j70s Green-j71s Red-j72s Yellow-j73s

with np.load('intersects_master.npz') as d:
    
    #First: which intersection, out of 6 (or 28)
    #Second: x,y
    #Third: j70s-j73s
    half=d['half']
    lin=d['lin']
    par=d['par']
    cub=d['cub']
    
    #Ordering of array parameters:
    #First: parameters for fit fn (m/b, a/b/c, a/b/c/d)
    #Second: Spike (horizontals, then verticals: 4 per image, or 8 halves)
    #Third: Image (j70s-j73s)
    pl=d['pl']
    pp=d['pp']
    pc=d['pc']
    plh=d['plh']


fig=plt.figure()
ax=fig.add_subplot(221)


#Plot half-spike linear fits
for i in range(8):
    if i<4:
        plt.plot(inds,line(inds,*plh[:,i,0]),'b-')
        plt.plot(inds,line(inds,*plh[:,i,1]),'g-')
        plt.plot(inds,line(inds,*plh[:,i,2]),'r-')
        plt.plot(inds,line(inds,*plh[:,i,3]),'y-')
    else:
        plt.plot(line(inds,*plh[:,i,0]),inds,'b-')
        plt.plot(line(inds,*plh[:,i,1]),inds,'g-')
        plt.plot(line(inds,*plh[:,i,2]),inds,'r-')
        plt.plot(line(inds,*plh[:,i,3]),inds,'y-')
plt.plot(half[:,0,0],half[:,1,0],'bo')
plt.plot(half[:,0,1],half[:,1,1],'go')
plt.plot(half[:,0,2],half[:,1,2],'ro')
plt.plot(half[:,0,3],half[:,1,3],'yo')

plt.title('Half-Spike Linear Fits')
plt.xlim([0,2462])
plt.ylim([0,2462])

ax=fig.add_subplot(222)

#Plot full-spike linear fits (likely a poor choice)
for i in range(4):
    if i<2:
        plt.plot(inds,line(inds,*pl[:,i,0]),'b-')
        plt.plot(inds,line(inds,*pl[:,i,1]),'g-')
        plt.plot(inds,line(inds,*pl[:,i,2]),'r-')
        plt.plot(inds,line(inds,*pl[:,i,3]),'y-')
    else:
        plt.plot(line(inds,*pl[:,i,0]),inds,'b-')
        plt.plot(line(inds,*pl[:,i,1]),inds,'g-')
        plt.plot(line(inds,*pl[:,i,2]),inds,'r-')
        plt.plot(line(inds,*pl[:,i,3]),inds,'y-')
plt.plot(lin[:,0,0],lin[:,1,0],'bo')
plt.plot(lin[:,0,1],lin[:,1,1],'go')
plt.plot(lin[:,0,2],lin[:,1,2],'ro')
plt.plot(lin[:,0,3],lin[:,1,3],'yo')

plt.title('Full-Spike Linear Fits')
plt.xlim([0,2462])
plt.ylim([0,2462])

ax=fig.add_subplot(223)

#Plot full-spike quadratic fits
for i in range(4):
    if i<2:
        plt.plot(inds,parab(inds,*pp[:,i,0]),'b-')
        plt.plot(inds,parab(inds,*pp[:,i,1]),'g-')
        plt.plot(inds,parab(inds,*pp[:,i,2]),'r-')
        plt.plot(inds,parab(inds,*pp[:,i,3]),'y-')
    else:
        plt.plot(parab(inds,*pp[:,i,0]),inds,'b-')
        plt.plot(parab(inds,*pp[:,i,1]),inds,'g-')
        plt.plot(parab(inds,*pp[:,i,2]),inds,'r-')
        plt.plot(parab(inds,*pp[:,i,3]),inds,'y-')
plt.plot(par[:,0,0],par[:,1,0],'bo')
plt.plot(par[:,0,1],par[:,1,1],'go')
plt.plot(par[:,0,2],par[:,1,2],'ro')
plt.plot(par[:,0,3],par[:,1,3],'yo')

plt.title('Quadratic Fits')
plt.xlim([0,2462])
plt.ylim([0,2462])

ax=fig.add_subplot(224)

#Plot full-spike cubic fits
for i in range(4):
    if i<2:
        plt.plot(inds,cubic(inds,*pc[:,i,0]),'b-')
        plt.plot(inds,cubic(inds,*pc[:,i,1]),'g-')
        plt.plot(inds,cubic(inds,*pc[:,i,2]),'r-')
        plt.plot(inds,cubic(inds,*pc[:,i,3]),'y-')
    else:
        plt.plot(cubic(inds,*pc[:,i,0]),inds,'b-')
        plt.plot(cubic(inds,*pc[:,i,1]),inds,'g-')
        plt.plot(cubic(inds,*pc[:,i,2]),inds,'r-')
        plt.plot(cubic(inds,*pc[:,i,3]),inds,'y-')
plt.plot(cub[:,0,0],cub[:,1,0],'bo')
plt.plot(cub[:,0,1],cub[:,1,1],'go')
plt.plot(cub[:,0,2],cub[:,1,2],'ro')
plt.plot(cub[:,0,3],cub[:,1,3],'yo')

plt.title('Cubic Fits')
plt.xlim([0,2462])
plt.ylim([0,2462])
plt.show()
