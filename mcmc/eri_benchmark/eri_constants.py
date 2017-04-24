def constants():

    import numpy as np

    distance = 29.43                        #distance to 51 eri
    parallax = 1./distance                  
    mstar = 1.75                            # mass of 51 eri from Simon + Schaefer 2011
    kgauss    =  np.double(0.017202098950) # Gauss's constant for orbital 
                                           # motion ... (kp)^2 = (2pi)^2 a^3
    
    CLIGHT    = np.double(2.99792458e10)   # speed of light
    AU        = 499.004782*CLIGHT          # the astronomical unit
    DAY       = np.double(8.64e4)	   # seconds

    return parallax,distance,mstar,kgauss,AU,DAY
