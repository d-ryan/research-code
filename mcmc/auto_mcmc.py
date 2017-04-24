import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
from orbitclass import orbit


def obsgen(elems,dist,times,err=20):
    '''
    Generate artificial observations with Gaussian errors from a set of orbital elements,
    distance (for AU->mas), and observation times.
    '''
    xs,ys = kepler(times,*elems)*1000./dist
    xs+=np.random.normal(0,err,xs.shape)
    ys+=np.random.normal(0,err,ys.shape)
        
    return xs,ys
    
def timegen(elems,t0,n=None,dt=365):
    '''
    Generate sequence of observation times based on desired parameters:
    n = number of observations
    t0 = time of first observation
    dt = spacing of observations :: if a scalar, uniform spacing. If a list of length
         n-1, allows for variable spacing between observations.
    '''
    times = [t0]
    if n is None:
        if dt is not None:
            for d in dt:
                times.append(times[-1]+d)
    else:
        if isinstance(dt,list):
            assert len(dt)==n-1
            for d in dt:
                times.append(times[-1]+d)        
        else:
            for i in range(n-1):
                times.append(times[-1]+dt)
    return times
    

###METHOD 1: Use old Kepler function
###METHOD 2: Use orbitclass.orbit(a=a,e=ecc,i=inc,lan=lan,argp=argp,tau=tau,m=mass).carttime(times)
###          This can be accessed readily via orbitclass.make_orbit(chain,mass).carttime(times)
def kepler(times,a,tau,argp,lan,inc,ecc,mass,dist):
    '''
    Return x,y coordinates at epochs, given orbital elements.
    
    Note that the binary-star/exoplanet orbital element convention is strange.
    +x is east (increasing R.A.), +y is north (increasing decl.),
    and +z is directly away from earth (increasing dist.).
    
    Longitude of the ascending node is defined by counterclockwise rotation
    (from Earth's perspective) from +y TOWARD +x.
    
    Inclination indicates rotation of the orbital plane about the line of nodes.
    If the ascending node represents the "top" and the descending node the "bottom"
    of the orbital plane, a POSITIVE inclination represents the LEFT side of the
    orbital plane (the portion b/w ascending -> descending node) being PUSHED INTO
    the plane of the sky, and the RIGHT side (portion b/w descending -> ascending)
    being LIFTED OUT of the sky plane.
    
    Argument of periapsis is again a counterclockwise rotation, in the same sense
    that longitude of the ascending node was.
    '''
    from etest import ema
    import numpy as np 
    
    from fmh_constants import constants

    # Epochs in MJD
    # 
    # date twice but x & y are computed.  
    # The data are returned so that the values in the array
    # alternate x and y pairs.
    #
    # Stellar properties for beta Pic
    parallax,distance,mstar,kgauss,AU,DAY = constants()
    ndate = len(epochs)
    
    distance = dist
    mstar = mass
    # Keplerian Elements 
    #
    # epochs      dates in modified JD [day]
    # a           semimajor axis [au]
    # tau         epoch of peri in units of the orbital period
    # argp        argument of peri [radians]
    # lan         longitude of ascending node [radians]
    # inc         inclination [radians]
    # ecc         eccentricity 
    # mass        total system mass
    # dist        distance from earth
    #
    # Derived quantities 
    # manom   --- mean anomaly
    # eccanom --- eccentric anomaly
    # truan   --- true anomaly
    # theta   --- longitude 
    # radius  --- star-planet separation
    n = kgauss*np.sqrt(mstar)*(a)**(-1.5)  # compute mean motion in rad/day
    # ---------------------------------------
    # Compute the anomalies (all in radians)
    # manom = n * (epochs - tau) # mean anomaly 
    manom = n*times - 2*np.pi*tau  # Mean anomaly w/tau in units of period
    eccanom = np.array([ema(man%(2.*np.pi),ecc) for man in manom])
    # ---------------------------------------
    # compute the true  anomaly and the radius
    # Elliptical orbit only
    truan = 2.*np.arctan(np.sqrt( (1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eccanom) )
    theta = truan + argp
    radius = a * (1.0 - ecc * np.cos(eccanom))
    # ---------------------------------------
    # Compute the vector components in 
    # ecliptic cartesian coordinates (normal convention Green Eq. 7.7)
    # standard form
    # xp = radius *(np.cos(theta)*np.cos(lan) - np.sin(theta)*np.sin(lan)*np.cos(inc))
    # yp = radius *(np.cos(theta)*np.sin(lan) + np.sin(theta)*np.cos(lan)*np.cos(inc))
    # write in terms of \Omega +/- omega --- see Mathematica notebook trig-id.nb
    c2i2 = np.cos(0.5*inc)**2
    s2i2 = np.sin(0.5*inc)**2
    arg0 = truan + lan
    arg1 = truan + argp + lan
    arg2 = truan + argp - lan
    arg3 = truan - lan
    c1 = np.cos(arg1)
    c2 = np.cos(arg2)
    s1 = np.sin(arg1)
    s2 = np.sin(arg2)
    sa0 = np.sin(arg0)
    sa3 = np.sin(arg3)
    
    # updated sign convention for Green Eq. 19.4-19.7
    xp = radius*(c2i2*s1 - s2i2*s2)
    yp = radius*(c2i2*c1 + s2i2*c2)

    # Interleave x & y
    # put x data in odd elements and y data in even elements
    data = np.zeros(ndate)
    data[0::2]  = xp
    data[1::2]  = yp

    return data*parallax # results in seconds of arc