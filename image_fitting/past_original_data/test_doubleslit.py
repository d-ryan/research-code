import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def double_slit(theta,d,b,lam):
    fac1 = np.cos(np.pi*d*np.sin(theta)/lam)**2.
    fac2 = np.sinc(np.pi*b*np.sin(theta)/lam)**2.
    return fac1*fac2
    
def theta(x,y):
    return np.arctan(x/y)
    
def fac1(theta,d,lam):
    return np.cos(np.pi*d*np.sin(theta)/lam)**2.
    
def fac2(theta,b,lam):
    return np.sinc(np.pi*b*np.sin(theta)/lam)**2.
    
coords = np.linspace(-1.,1.,10001)
y = 3.
d = 5e-4
b = 1e-4
lam = 6e-7

#BELOW: FITTING FUNCTIONS    
    
def dso(x,cen,o,s1,s2,c,A):
    '''cen: center of diffraction pattern
    o: offset between diffraction center and double-slit peak
    s1: width of peak (cos)
    s2: width of envelope (sinc)
    c: additive offset
    A: normalization
    '''
    
    t1 = np.cos(s1*(x-cen-o))**2.
    t2 = np.sinc(s2*(x-cen))**2.
    return A*t1*t2+c
    
def dsreg(x,cen,s1,s2,c,A):
    t1 = np.cos(s1*(x-cen))**2.
    t2 = np.sinc(s2*(x-cen))**2.
    return A*t1*t2+c
    
def gauss(x,A,mu,sig,o):
    '''
    gaussian with width sig, mean mu, normalization A and additive offset o
    '''
    norm = A/np.sqrt(2.*np.pi)/sig
    expon = ((x-mu)**2.)/(2.*sig**2.)
    return norm*np.exp(-expon)
    
def sinc2(x,a,p,s,o):
    '''
    np.sinc(x) = sin(pi*x)/(pi*x)
    this sinc2 fn uses 's' as a multiplicative width (small s = wider),
    'p' as a left-right offset, 'a' as a multiplicative prefactor.
    These values are then squared, and a vertical offset 'o' added.
    '''
    f=np.sinc(s*(x-p))
    return a*f**2+o
    
def threesinc2(x,a1,p1,s2,a2,p2,s2,a3,p3,s3,o):
    '''
    Additive function with 3 sinc^2 profiles for doubled spikes (ie WFPC2)
    Contains only a single offset term to prevent degeneracy
    '''
    return sinc2(x,a1,p1,s1,o/3.)+sinc2(x,a2,p2,s2,o/3.)+sinc2(x,a3,p3,s3,o/3.)