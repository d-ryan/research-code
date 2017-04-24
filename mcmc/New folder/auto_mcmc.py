import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv


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
    
def kepler(times,a,tau,argp,lan,inc,ecc):
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
    