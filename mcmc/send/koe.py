### 
# This function accepts keplarian orbital elements and returns x,y positions
###



def koe(epochs,a,tau,argp,lan,inc,ecc):

    from etest import ema
    import numpy as np 
    
    from eri_constants import constants

    # Epochs in MJD
    # 
    # date twice but x & y are computed.  
    # The data are returned so that the values in the array
    # alternate x and y pairs.
    #
    # Stellar properties for beta Pic

    parallax,distance,mstar,kgauss,AU,DAY = constants()
                                       
    ndate = len(epochs)

    # Keplerian Elements 
    #
    # epochs      dates in modified JD [day]
    # a           semimajor axis [au]
    # tau         epoch of peri in units of the orbital period
    # argp        argument of peri [radians]
    # lan         longitude of ascending node [radians]
    # inc         inclination [radians]
    # ecc         eccentricity 
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
    #
    # manom = n * (epochs - tau) # mean anomaly 

    manom = n*epochs[0::2] - 2*np.pi*tau  # Mean anomaly w/tau in units of period

    eccanom = np.array([])
 
    for man in manom:
        #print man #DEBUGGING
        eccanom = np.append(eccanom, ema(man%(2*np.pi), ecc))

    # ---------------------------------------
    # compute the true  anomaly and the radius
    #
    # Elliptical orbit only

    truan = 2.*np.arctan(np.sqrt( (1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eccanom) )
    theta = truan + argp
    radius = a * (1.0 - ecc * np.cos(eccanom))

    # ---------------------------------------
    # Compute the vector components in 
    # ecliptic cartesian coordinates (normal convention Green Eq. 7.7)
    # standard form
    #
    #xp = radius *(np.cos(theta)*np.cos(lan) - np.sin(theta)*np.sin(lan)*np.cos(inc))
    #yp = radius *(np.cos(theta)*np.sin(lan) + np.sin(theta)*np.cos(lan)*np.cos(inc))
    #
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

