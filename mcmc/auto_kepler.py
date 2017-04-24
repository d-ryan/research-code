import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
from orbitclass import orbit

G = 4.*np.pi**2.

days_sidereal_year = 365.25636
dsy = days_sidereal_year

    

def koe(epochs,q,tau,argp,lan,inc,ecc,mass,dist):
    #define mean motion
    
    if ecc<0: raise ValueError('Eccentricity cannot be negative')
    elif ecc==0:
        a = q
        n = np.sqrt(G*mass/a**3.)/dsy #units: 2pi/P in YEARS
        M = n*epochs-2.*np.pi*tau
        theta = M%(2.*np.pi)
    elif ecc<1:
        a = q/(1.-ecc)
        n = np.sqrt(G*mass/a**3.)/dsy
        M = n*epochs-2.*np.pi*tau
        theta = M2nue(M,ecc)
    elif ecc==1:
        n = np.sqrt(G*mass/(2.*q**3.))/dsy
        M = n*epochs-2.*np.pi*tau
        theta = M2nup(M)
    else:
        a = q/(1.-ecc)
        n = np.sqrt(G*mass/np.abs(a)**3.)/dsy
        M = n*epochs-2.*np.pi*tau
        theta = M2nuh(M,ecc)
    
    l_half=q*(1.+ecc)
    r = l_half/(1.+ecc*np.cos(theta))
    
    sO=np.sin(lan)
    cO=np.cos(lan)
    si=np.sin(inc)
    ci=np.cos(inc)
    sw=np.sin(argp+theta)
    cw=np.cos(argp+theta)
    x=r*(sO*cw+cO*ci*sw)/dist
    y=r*(cO*cw-sO*ci*sw)/dist
    return x,y
    
def M2nue(M,ecc):
    M = M%(2*np.pi)
    if np.size(M)>1: #Can work on arrays of anomalies with this syntax
        flag = np.zeros(M.shape)
        flag[M>np.pi]=1
        M[flag==1]=2.*np.pi-M[flag==1]
    else: #Or on individual anomalies with THIS syntax
        flag=False
        if M > np.pi:
            flag = True
            M = 2.*np.pi - M
    alpha = (1. - ecc)/(4.*ecc + 0.5)
    beta = 0.5*M / (4. * ecc + 0.5)
    aux = np.sqrt(beta**2. + alpha**3.)
    z = beta + aux
    z = z**(1./3.)
    s0 = z - (alpha/z)
    s1 = s0 - (0.078*(s0**5.0)) / (1.0 + ecc)
    e0 = M + (ecc * (3.0*s1 - 4.0*(s1**3.0)))
    se0 = np.sin(e0)
    ce0 = np.cos(e0)
    f = e0-ecc*se0-M
    f1 = 1.0-ecc*ce0
    f2 = ecc*se0
    f3 = ecc*ce0
    f4 = -f2
    u1 = -f/f1
    u2 = -f/(f1+0.5*f2*u1)
    u3 = -f/(f1+0.5*f2*u2+0.1666666666667*f3*u2*u2)
    u4 = -f/(f1+0.5*f2*u3+0.1666666666667*f3*u3*u3+0.04166666666667*f4*(u3**3.0))    
    E = (e0 + u4)
    if np.size(M)>1: #Array case
        M[flag==1]=2.*np.pi-M[flag==1]
        E[flag==1]=2.*np.pi-E[flag==1]
    else: #individual case
        if flag:
            M = 2.*np.pi - M
            E = 2.*np.pi - E
    theta=np.arccos((np.cos(E)-ecc)/(1.-ecc*np.cos(E)))
    if np.size(E)>1:
        theta[E>np.pi]=2.*np.pi-theta[E>np.pi]
    else:
        if E>np.pi:
            theta=2.*np.pi-theta
    return theta

def M2nup(M):
    z = (3./2.*M+np.sqrt(9./4.*M**2.+1))**(1./3.)
    return 2.*np.arctan(z-1./z)

def M2nuh(M,ecc):
    alpha = (ecc-1.)/(4.*ecc+0.5)
    beta = 0.5*M/(4.*ecc+0.5)
    aux = np.sqrt(beta**2. + alpha**3.)
    if np.size(M)>1:
        z = beta
        z[beta>=0]+=aux[beta>=0]
        z[beta>=0]**=(1./3.)
        z[beta<0]-=aux[beta<0]
        z[beta<0] = -(-z[beta<0])**(1./3.)
    else:
        if beta >= 0:
            z = beta + aux
            z**=(1./3.)
        else:
            z = beta - aux
            z = -(-z)**(1./3.)
    s0 = z - (alpha/z)
    s1 = s0 + 0.071*s0**5./((1.+0.45*s0**2.)*(1.+4.*s0**2.)*ecc)
    e0 = 3*np.log(s1 + np.sqrt(1. + s1**2.))
    f2=f4=ecc*np.sinh(e0)
    f3=ecc*np.cosh(e0)
    f1=f3-1.
    f=f2-e0-M
    u1 = -f/f1
    u2 = -f/(f1+0.5*f2*u1)
    u3 = -f/(f1+0.5*f2*u2+0.1666666666667*f3*u2*u2)
    u4 = -f/(f1+0.5*f2*u3+0.1666666666667*f3*u3*u3+0.04166666666667*f4*(u3**3.0))    
    E = (e0 + u4)
    theta = np.arccos((ecc-np.cosh(E))/(ecc*np.cosh(E)-1.))
    if np.size(E)>1:
        theta[E<0]*=-1.
    else:
        if E<0:
            theta*=-1.
    return theta