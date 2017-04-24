import numpy as np

def rob(Manom,e):
    flag = False
    if Manom > np.pi:
        flag = True
        Manom = 2.*np.pi - Manom
    alpha = (1. - e)/(4.*e + 0.5)
    beta = 0.5*Manom / (4. * e + 0.5)
    aux = np.sqrt(beta**2. + alpha**3.)
    z = beta + aux
    z = z**(1./3.)
    
    if e == 0:
        return Manom

    s0 = z - (alpha/z)
    
    s1 = s0 - (0.078*(s0**5.0)) / (1.0 + e)
    e0 = Manom + (e * (3.0*s1 - 4.0*(s1**3.0)))
    
    se0 = np.sin(e0)
    ce0 = np.cos(e0)
    
    f = e0-e*se0-Manom
    f1 = 1.0-e*ce0
    f2 = e*se0
    f3 = e*ce0
    f4 = -f2
    u1 = -f/f1
    u2 = -f/(f1+0.5*f2*u1)
    u3 = -f/(f1+0.5*f2*u2+0.1666666666667*f3*u2*u2)
    u4 = -f/(f1+0.5*f2*u3+0.1666666666667*f3*u3*u3+0.04166666666667*f4*(u3**3.0))
    
    Eanom = (e0 + u4)
    if flag == True:
        Manom = 2.*np.pi - Manom
        Eanom = 2.*np.pi - Eanom
    return Eanom

def Eguess(M,e):
    alpha = (1. - e)/(4.*e+0.5)
    beta = M/2./(4.*e+0.5)

    if e == 0:
        return M
    elif e < 1:
        temp = beta + np.sign(beta)*np.sqrt(beta**2.+alpha**3.)
        if beta < 0:
            z = -(-temp)**(1./3.)
        else:
            z = temp**(1./3.)
        s = z - alpha/z
        s -= 0.078*(s**5.)/(1.+e)
        return M + e*(3.*s-4*s**3.)
    elif e == 1:
        print "Parabola not implemented"
        return
    elif e > 1:
        print 'hyperbola'
        alpha *= -1.
        temp = beta + np.sign(beta)*np.sqrt(beta**2.+alpha**3.)
        if beta < 0:
            z = -(-temp)**(1./3.)
        else:
            z = temp**(1./3.)
        s = z - alpha/z
        s += 0.071*(s**5.)/((1.+0.45*s**2.)*(1.+4.*s**2.)*e)
        return 3.*np.log(s+np.sqrt(1.+s**2.))
    else:
        print "Your e is weird"
        return

def correction(E,M,e):
    if e == 0:
        return E
    elif e < 1:
        sE=np.sin(E)
        cE=np.cos(E)
        f2 = e*sE
        f3 = e*cE
        f1 = 1. - f3
        f = E - f2 - M
        f4 = -f2
    elif e > 1:
        print 'hyperbola'
        sE=np.sinh(E)
        cE=np.cosh(E)
        f2 = e*sE
        f3 = e*cE
        f1 = f3 - 1.
        f = f2 - E - M
        f4 = f2
    else:
        print "Your e is weird"
        return
    u1 = -f/f1
    u2 = -f/(f1 + .5*f2*u1)
    u3 = -f/(f1 + .5*f2*u2 + f3*(u2**2.)/6.)
    u4 = -f/(f1 + .5*f2*u3 + f3*(u3**2.)/6. + f4*(u3**3.)/24.)
    return E + u4

    
#vectorized functions: allows them to be called on numpy arrays elementwise 
vEguess = np.vectorize(Eguess)        
vcorrection = np.vectorize(correction)
vrob = np.vectorize(rob)



def ema(M,e):
    return rob(M,e)

##### Stuff for testing: elliptical case


#generate some eccentric anomalies to test
Etest=np.linspace(-np.pi,np.pi,1000)

#for a variety of eccentricities
etest=np.linspace(0,.999,1000)

#make matrices to explore the full -pi < E < pi and 0 < e < 1 parameter space
ones=np.ones(1000)
etest=np.outer(etest[:],ones)
Etest=np.outer(ones,Etest[:])

#calculate mean anomalies
Mtest=Etest-etest*np.sin(Etest)

#guess the ecc. anomaly that produced these mean anomalies
E0=vEguess(Mtest,etest)
E1=vcorrection(E0,Mtest,etest)

Er=vrob(Mtest,etest)

uncmax = np.max(np.abs((E0-Etest)/Etest)) #largest uncorrected error
cormax = np.max(np.abs((E1-Etest)/Etest)) #largest corrected error

rmax = np.max(np.abs((Er-Etest)/Etest)) #largest error doing what rob does--equivalent to mine,
 #but uses different angle ranges


####### Devnotes

# 6-2: code runs extremely well for e<1; need to incorporate with the previous codes and make
#something that runs smoothly for all e. (max error was 10^-13, possibly machine-limited)

# Need also to test agains EccAn_rnu_PT.py
