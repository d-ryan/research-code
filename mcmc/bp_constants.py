def constants():

    import numpy as np

    parallax  = np.double(51.44e-3)        # parallax of Beta Pic - from Chauvin
    distance  = np.double(1./parallax)     # parsec
    mstar     = np.double(1.75e0)          # mass of beta Pic from Chauvin
    kgauss    =  np.double(0.017202098950) # Gauss's constant for orbital 
                                           # motion ... (kp)^2 = (2pi)^2 a^3
    
    CLIGHT    = np.double(2.99792458e10)   # speed of light
    AU        = 499.004782*CLIGHT          # the astronomical unit
    DAY       = np.double(8.64e4)	   # seconds

    return parallax,distance,mstar,kgauss,AU,DAY
