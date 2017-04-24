#anomaly calculator benchmarking tool

import numpy as np
import matplotlib.pyplot as plt
import time

start = time.clock()

Etest=np.linspace(-np.pi,np.pi,1000)
etest=np.linspace(0,.999,1000)
ones=np.ones(1000)
etest=np.outer(etest[:],ones)
Etest=np.outer(ones,Etest[:])
Mtest=Etest-etest*np.sin(Etest)
times_rob = np.zeros(Mtest.shape) # Time to perform operation 5x.
times_me = times_rob.copy()
times_g1 = times_rob.copy()
times_g2 = times_rob.copy()
times_g3 = times_rob.copy()
times_g4 = times_rob.copy()
times_g5 = times_rob.copy()

i1 = times_rob.copy()
i2 = i1.copy()
i3 = i1.copy()
i4 = i1.copy()
i5 = i1.copy()

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

def g1(M,e,true):
    #attempt 1:
    #Guess E = (complicated)
    guess = M + e*np.sin(M)/(1-np.sin(M+e)+np.sin(M))
    return newton(M,e,guess,true)
    
def g2(M,e,true):
    #attempt 2:
    #Guess E = M
    return newton(M,e,M,true)
    
def g3(M,e,true):
    #attempt 3:
    guess = M + np.sign(M)*e
    return newton(M,e,guess,true)
    
def g4(M,e,true):
    #attempt 4:
    guess = M + e*M
    return newton(M,e,guess,true)
    
def g5(M,e,true):
    #attempt 5:
    guess = M+e*np.sin(M)
    return newton(M,e,guess,true)
        
        
def newton(M,e,guess,true):
    max_err = 1e-9
    err=np.abs(true-guess)
    max_iter = 1000
    i=0
    
    while err>max_err:
        if i>=max_iter: break
        dE = (M-guess+e*np.sin(guess))/(1.-e*np.cos(guess))
        guess+=dE
        err=np.abs(true-guess)
        i+=1
    return i,guess
        

    
#vectorized functions: allows them to be called on numpy arrays elementwise 
vEguess = np.vectorize(Eguess)        
vcorrection = np.vectorize(correction)
vrob = np.vectorize(rob)

def myfun(M,e):
    E = Eguess(M,e)
    return correction(E,M,e)

def bench(M,e,fun,trueval=None,):
    tstart = time.clock()
    if trueval is not None:
        for k in range(5):
            i,temp = fun(M,e,trueval)
        tend=time.clock()-start
        return i,tend
    else:
        for k in range(5):
            temp = fun(M,e)
        tend = time.clock()-tstart
        return tend


for i in range(1000):
    for j in range(1000):
        M = Mtest[i,j]
        e = etest[i,j]
        times_rob[i,j]=bench(M,e,rob)
        times_me[i,j]=bench(M,e,myfun)
        i1[i,j],times_g1[i,j]=bench(M,e,g1,Etest[i,j])
        i2[i,j],times_g2[i,j]=bench(M,e,g2,Etest[i,j])
        i3[i,j],times_g3[i,j]=bench(M,e,g3,Etest[i,j])
        i4[i,j],times_g4[i,j]=bench(M,e,g4,Etest[i,j])
        i5[i,j],times_g5[i,j]=bench(M,e,g5,Etest[i,j])

#np.savez_compressed('ema_benchmark',tr=ema_benchmark.times_rob,td=ema_benchmark.times_me,t1=ema_benchmark.times_g1,t2=ema_benchmark.times_g2,t3=ema_benchmark.times_g3,t4=ema_benchmark.times_g4,t5=ema_benchmark.times_g5,i1=ema_benchmark.i1,i2=ema_benchmark.i2,i3=ema_benchmark.i3,i4=ema_benchmark.i4,i5=ema_benchmark.i5)

        #ema_benchmark.
'''
#guess the ecc. anomaly that produced these mean anomalies
E0=vEguess(Mtest,etest)
E1=vcorrection(E0,Mtest,etest)
E2=vema(Mtest,etest)

Er=vrob(Mtest,etest)

uncmax = np.max(np.abs((E0-Etest)/Etest)) #largest uncorrected error
cormax = np.max(np.abs((E1-Etest)/Etest)) #largest corrected error
emamax = np.max(np.abs((E2-Etest)/Etest)) #largest Newton error

rmax = np.max(np.abs((Er-Etest)/Etest))

uncmean = np.mean(np.abs((E0-Etest)/Etest)) #largest uncorrected error
cormean = np.mean(np.abs((E1-Etest)/Etest)) #largest corrected error
emamean = np.mean(np.abs((E2-Etest)/Etest)) #largest Newton error

rmean = np.mean(np.abs((Er-Etest)/Etest))


print uncmean
print cormean
print rmean
print emamean
'''