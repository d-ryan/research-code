import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import jdcal
from mpl_toolkits.mplot3d import Axes3D

def make_orbit(chain,mass):
    a=np.exp(chain[0])
    e=chain[5]
    i=np.arccos(chain[4])
    lan=chain[3]
    argp=chain[2]
    P=np.sqrt(a**3./mass)*365.256
    tau=(chain[1]*P+50000-49718)%P+49718
    orb=orbit(a=a,e=e,i=i,lan=lan,argp=argp,tau=tau,m=mass)
    return orb

def plot_from_chain(chain,mass,dist=None,filename=None):
    orb = make_orbit(chain,mass)
    if dist is not None:
        if filename is None:
            print 'Both distance and filename needed to plot observations'
            orb.plot_orbit()
            return
        orb.plot_orbit(obs=orb.get_obs(filename,dist))
        return
    if filename is not None:
        print 'Both distance and filename needed to plot observations'
    orb.plot_orbit()
    return
    
class orbit(object):
    #-------------------------------------------------------------
    #Startup methods: __init__ and value-setters
    #(so e.g. mass guesses can be adjusted later)
    # NOTE: adjust this to accept q's and be valid for parabolic & hyperbolic orbits too
    #-------------------------------------------------------------
    def __init__(self,a=1,e=0,i=0,lan=0,argp=0,m=None,tau=0,P=1,m_planet=None,sky=True,id=None,dist=None):
        self.set_geometry(a,e,i,lan,argp)
        self.set_tau(tau)
        self.set_sky(sky)
        self.set_mass_period(m,P)
        self.set_planet_mass(m_planet)
        self.set_id(id)
        self.set_dist(dist)
    def set_geometry(self,a,e,i,lan,argp):
        self.a = a #semimajor axis in AU
        self.e = e #eccentricity
        self.i = i #inclination (radians)
        self.lan = lan #longitude of ascending node (radians)
        self.O = lan #LAN = Big Omega
        self.argp = argp #argument of periapsis (radians)
        self.w = argp #AoP = little omega (looks like w)
        #self.tau = tau 
        #self.sky = sky 
    def set_tau(self,tau):
        if tau==0:
            print 'Using default tau of MJD 0'
        self.tau = tau #epoch of periapsis, in MJD
    def set_sky(self,sky):
        self.sky = sky #sky-plane (True) or solar-system (False)coordinate conventions?
    def set_mass_period(self,m,P):
        if m is not None:
            if P!=1:
                print 'Overspecified! Give exactly one of m and P'
                print 'Ignoring period input for this calculation'
            self.m = m #mass (solar masses)
            self.P = np.sqrt(self.a**3./m) #period (yrs)
        else:
            if P==1:
                print 'Using default period of 1 yr to calculate mass'
            self.m = self.a**3./P**2. #mass (solar masses)
            self.P = P #period (yrs)
        self.mu = self.m*4.*np.pi**2. #Standard grav. parameter
        self.h = np.sqrt(self.mu*self.a*(1.-self.e**2)) #specific relative angular momentum
    def set_planet_mass(self,m_planet):
        if m_planet is None:
            #print 'No planet mass given'
            self.m_planet=m_planet
        else:
            self.m_planet = 0.000954265748*m_planet #planet mass (give in jupiters, gets converted to solar)
            if self.m_planet > 0.05*self.m:
                print 'Planet more than 5% stellar mass! Accuracy in question!'
    def set_id(self,id):
        self.id=id
    def set_dist(self,dist):
        self.dist=dist


    #-------------------------------------------------------------
    #Distance methods: Peri and apo as hard-wired properties,
    #and radius (separation) at a given true anomaly
    #-------------------------------------------------------------
    @property
    def peri(self):
        '''Returns periapsis distance.'''
        return self.a*(1.-self.e)
    @property
    def apo(self):
        '''Returns apoapsis distance.'''
        return self.a*(1.+self.e)
    def r(self,theta):
        '''Returns separation of parent body and orbiting body.'''
        return self.a*(1.-self.e**2.)/(1.+self.e*np.cos(theta))
    
    
    
    #-------------------------------------------------------------
    #methods to interchange between
    #true, eccentric, and mean anomalies
    #as well as phases/time since periapsis/absolute times
    #-------------------------------------------------------------
    def ecc2true(self,E):
        '''Given eccentric anomaly, returns true anomaly.'''
        theta=np.arccos((np.cos(E)-self.e)/(1.-self.e*np.cos(E)))
        if np.size(E)>1: #can work on arrays with this syntax
            theta[E>np.pi]=2.*np.pi-theta[E>np.pi]
        else: #or on individual anomalies with THIS syntax
            if E>np.pi:
                theta=2.*np.pi-theta
        return theta
    def true2ecc(self,theta):
        '''Given true anomaly, returns eccentric anomaly.'''
        E = np.arccos((self.e+np.cos(theta))/(1.+self.e*np.cos(theta)))
        if np.size(theta)>1: #can work on arrays with this syntax
            E[theta>np.pi]=2.*np.pi-E[theta>np.pi]
        else: #or on individual anomalies with THIS syntax (note: worth making fn() to handle these?)
            if theta>np.pi:
                E = 2.*np.pi - E
        return E
    def ecc2mean(self,E):
        '''Given eccentric anomaly, returns mean anomaly.'''
        return E-self.e*np.sin(E)
    def mean2ecc(self,M):
        '''Given mean anomaly, returns eccentric anomaly.'''
        if self.e == 0: #No eccentricity? No problem
            return M
        if np.size(M)>1: #Can work on arrays of anomalies with this syntax
            flag = np.zeros(M.shape)
            flag[M>np.pi]=1
            M[flag==1]=2.*np.pi-M[flag==1]
        else: #Or on individual anomalies with THIS syntax
            flag=False
            if M > np.pi:
                flag = True
                M = 2.*np.pi - M
        alpha = (1. - self.e)/(4.*self.e + 0.5)
        beta = 0.5*M / (4. * self.e + 0.5)
        aux = np.sqrt(beta**2. + alpha**3.)
        z = beta + aux
        z = z**(1./3.)
        s0 = z - (alpha/z)
        s1 = s0 - (0.078*(s0**5.0)) / (1.0 + self.e)
        e0 = M + (self.e * (3.0*s1 - 4.0*(s1**3.0)))
        se0 = np.sin(e0)
        ce0 = np.cos(e0)
        f = e0-self.e*se0-M
        f1 = 1.0-self.e*ce0
        f2 = self.e*se0
        f3 = self.e*ce0
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
        return E
    def true2mean(self,theta):
        '''Given true anomaly, returns mean anomaly.'''
        E = self.true2ecc(theta)
        return self.ecc2mean(E)
    def mean2true(self,M):
        '''Given mean anomaly, returns true anomaly.'''
        E = self.mean2ecc(M)
        return self.ecc2true(E)
    def phase2true(self,ph):
        '''Given phase, returns true anomaly.'''
        return self.mean2true(2.*np.pi*(ph%1.))
    def true2phase(self,theta):
        '''Given true anomaly, returns orbital phase.'''
        return self.true2mean(theta)/2./np.pi
    def dt2true(self,dt):
        '''Given time since periapsis in years, returns true anomaly.'''
        return self.mean2true(2.*np.pi*((dt/self.P)%1.))
    def true2dt(self,theta):
        '''Given true anomaly, returns time since periapsis in years.
        Restricted to one period after the known epoch of periapsis.'''
        return self.true2mean(theta)/2./np.pi*self.P
    def time2true(self,t):
        '''Given absolute time [MJD], returns true anomaly.'''
        return self.mean2true(2.*np.pi*(((t-self.tau)/(self.P*365.256))%1.))
    def true2time(self,theta):
        '''Given true anomaly, returns absolute time [MJD].
        Restricted to one period after the known epoch of periapsis.'''
        return self.true2mean(theta)/2./np.pi*self.P*365.256+self.tau
    
    
    #-------------------------------------------------------------
    # Coordinate methods: (x,y,z,vx,vy,vz) according to sky-plane conventions
    # Note: Each individual-coordinate method calls cartesian(), which calculates
    #       all six coordinates. Avoid using these and use cartesian() instead,
    #       whenever multiple coorinates or velocities are needed.
    #       The methods x(), y(), etc. are for convenience, not efficiency.
    #-------------------------------------------------------------
    
    def x(self,theta):
        '''Returns x-offset (west) of orbiting body, given true anomaly.'''
        return self.cartesian(theta)[0]
    def y(self,theta):
        '''Returns y-offset (north) of orbiting body, given true anomaly.'''
        return self.cartesian(theta)[1]
    def z(self,theta):
        '''Returns z-offset (line of sight) of orbiting body, given true anomaly.'''
        return self.cartesian(theta)[2]
    def vx(self,theta):
        '''Returns x-velocity of orbiting body, given true anomaly.'''
        return self.cartesian(theta)[3]
    def vy(self,theta):
        '''Returns y-velocity of orbiting body, given true anomaly.'''
        return self.cartesian(theta)[4]
    def vz(self,theta):
        '''Returns z-velocity of orbiting body, given true anomaly.'''
        return self.cartesian(theta)[5]
    def carttime(self,t):
        return self.cartesian(self.time2true(t))
    #Double-check these formulae!!!
    def cartesian(self,theta):
        '''Returns (x,y,z,vx,vy,vz) for this orbit when given theta.
        Currently set up for sky-plane conventions (x east, y north)'''
        sO=np.sin(self.O)
        cO=np.cos(self.O)
        si=np.sin(self.i)
        ci=np.cos(self.i)
        sw=np.sin(self.w + theta)
        cw=np.cos(self.w + theta)
        r = self.r(theta)
        x = r*(sO*cw + cO*ci*sw)
        y = r*(cO*cw - sO*ci*sw)
        z = r*si*sw
        theta_dot = self.h/r**2.
        vx = theta_dot*(x*self.e*np.sin(theta)/(1.+self.e*np.cos(theta)) + r*(cO*ci*cw-sO*sw))
        vy = theta_dot*(y*self.e*np.sin(theta)/(1.+self.e*np.cos(theta)) - r*(sO*ci*cw+cO*sw))
        vz = theta_dot*(z*self.e*np.sin(theta)/(1.+self.e*np.cos(theta)) + r*si*cw)
        return x,y,z,vx,vy,vz

    #-------------------------------------------------------------
    # Reflex motion methods: reflex_motion() is just cartesian() for the star,
    # while reflex_signal() gives max amplitudes of certain observables
    #
    # DEFINITELY more work to be done here
    #
    #-------------------------------------------------------------        
    def reflex_motion(self,theta):
        '''Returns APPROXIMATE (x,y,z,vx,vy,vz) of star relative to barycenter.
        Calculated by (-1)*(m_planet/m_star)*(planet's x,y,z,vx,vy,vz)'''
        if self.m_planet is None:
            print 'Planet mass needed; use orbit.set_planet_mass(m_planet) to set'
            return
        data = self.cartesian(theta)
        return -self.m_planet/self.m*data
        
    def reflex_signal(self,phase=False,npts=10001):
        '''Returns APPROXIMATE max amplitude of:
        -radial velocity oscillation
        -sky-plane wobble: maximum and x-y components
        -phases at which bounding values are reached, if desired (NOT IMPLEMENTED)
        '''
        nus = np.linspace(0.,np.pi*2.,npts)
        data = -self.m_planet/self.m*np.array(self.cartesian(nus))
        vsig = np.amax(data[5])-np.amin(data[5])
        xsig = np.amax(data[0])-np.amin(data[0])
        ysig = np.amax(data[1])-np.amin(data[1])
        sep = np.sqrt(data[0]**2 + data[1]**2)
        maxima = self.local_maxima(sep)
        if len(maxima) != 2:
            print 'Too many maxima in sky-plane offset from barycenter'
            print 'Maximum offset from barycenter returned instead'
            print 'This should not be used as true maximum wobble'
            wobble = np.amax(sep)
        else:
            if np.amax(maxima) == np.amax(sep):
                wobble = maxima[0]+maxima[1]
            else:
                print 'Trouble calculating max offset from barycenter'
                print 'numpy max offset returned instead'
                print 'This should not be used as true maximum wobble'
                wobble = np.amax(sep)
        return vsig,xsig,ysig,wobble
    
    #maintenance method for reflex_signal -- NEEDS TESTING
    def local_maxima(self,array,ind=False):
        '''Return either values or indices of all local maxima of an array.'''
        vals=[]
        number=len(array)
        for i in range(number):
            if array[i]>array[i-1] and array[i]>array[(i+1)%number]:
                vals.append(i)
        if ind:
            return vals
        else:
            return array[vals]



    def get_plot_data(self,axis='-z',facing=0,npts=10001,tflag=False,z=False):
        nus = np.linspace(0.,np.pi*2.,npts)
        data = self.cartesian(nus)
        xflag,yflag=self.set_flags(axis,facing)
        x,y=self.use_flags(data,xflag,yflag)
        if z:
            z = -data[2]
            return x,y,z
        if tflag:
            t = self.true2time(nus)
            return x,y,t
        return x,y
    #-------------------------------------------------------------
    # Plotting methods: plot_orbit will draw the orbital trajectory
    # plus any given observed points (with errors--NOT IMPLEMENTED)
    # while also highlighting specified anomalies/times/phases
    # 
    # Also includes several maintenance/convenience methods for plotting
    #-------------------------------------------------------------
    def plot_orbit_3d(self,orientation='earth'):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        if orientation=='earth':
            x,y,z = self.get_plot_data(z=True)
        else:
            nus = np.linspace(0.,np.pi*2.,10001)
            data = self.cartesian(nus)
            x = data[0]
            y = data[1]
            z = data[2]
        plt.plot([0],[0],[0],'yo')
        ax.plot(x,y,z,'b-')
        ax.set_aspect('equal')
        xmax = np.max([np.abs(np.max(x)),np.abs(np.min(x))])
        ymax = np.max([np.abs(np.max(y)),np.abs(np.min(y))])
        zmax = np.max([np.abs(np.max(z)),np.abs(np.min(z))])
        MAX = np.max([xmax,ymax,zmax])
        for direction in (-1,1):
            for point in np.diag(direction*MAX*np.array([1,1,1])):
                ax.plot([point[0]],[point[1]],[point[2]],'w')
             
        plt.show()
    def plot_orbit(self,axis='-z',facing=0,ptlist=None,pttype='TA',npts=1001,obs=None,err=None):
        '''Plots an orbit, looking along a given axis, and highlights any desired points.
        axis = {'x','y','z','-x','-y','-z'} : refers to where the observer is relative to origin
        facing = {0, 1, 2, 3} : orientation of axes; if for any reason you want to turn it on its side
        pttype = 'TA' : True anomaly; 'EA' : eccentric, 'MA' : mean, 'TP' = time since peri, 'T' = julian dates, 'P' = phase
        ptlist = [val0,val1,val2,...] : specific time/phase/anomaly values at which to plot points
        npts = int : number of points with which to draw the full orbital path
        obs = [[x1,x2,x3,...],[y1,y2,y3,...]] Coordinates of observations, using +x: east (left) y: north (up)
        err = [[x1,x2,x3,...],[y1,y2,y3,...]] All must be positive. x & y errors in same format as previous.
        
        'Facing' is a terrible system, explained below:
        How facing works: 'axis' selects one of six viewpoints. Of the two remaining axes, the
        alphabetically LAST (y if xy, z else) is placed according to the 'facing' variable: 0 up, 1 rt, 2 dn, 3 lft.
        The final axis is chosen to complete a right-handed coordinate system.'''

        #make sure requested axis is a valid option
        options = np.array(['x','y','z','-x','-y','-z'])
        if not np.any(axis == options):
            print 'invalid axis'
            return
        
        #if points are requested, acquire x,y data for them
        ptflag=False
        if ptlist is not None:
            ptflag=True
            if pttype == 'TA': #true anomaly: easy
                ptdata=self.cartesian(ptlist)
            elif pttype == 'MA': #mean anomaly: convert to true
                temp = np.array([self.mean2true(i) for i in ptlist])
                ptdata=self.cartesian(temp)
            elif pttype == 'EA': #eccentric anomaly: convert to true
                temp = np.array([self.ecc2true(i) for i in ptlist])
                ptdata=self.cartesian(temp)
            elif pttype == 'TP': #time since peri: convert to mean, then true anomaly
                temp = np.array([self.dt2true(i) for i in ptlist])
                ptdata=self.cartesian(temp)
            elif pttype == 'T': #absolute time: subtract off epoch of peri, then as with 'TP'
                temp = np.array([self.time2true(i) for i in ptlist])
                ptdata=self.cartesian(temp)
            elif pttype == 'P': #phase: convert to mean anomaly more easily
                temp = np.array([self.phase2true(i) for i in ptlist])
                ptdata=self.cartesian(temp)
            else:
                print 'invalid style of point'
                print 'please choose from:'
                print 'TA MA EA TP T P'
        
        #turn the requested axis & orientation (0,1,2,3) into useful information
        xflag,yflag=self.set_flags(axis,facing)
        
        #set up data to plot: trajectory, highlighted points (if requested), observed points (if given)
        ######x,y=self.use_flags(data,xflag,yflag)
        x,y = self.get_plot_data(axis,facing,npts)
        if ptflag:
            xpts,ypts=self.use_flags(ptdata,xflag,yflag)
        if obs is not None:
            if axis == '-z':
                xobs,yobs=self.use_flags(obs,xflag,yflag)
                plt.plot(xobs,yobs,'ro') #plot observed points
            else:
                print 'Cannot plot observations from a perspective other than Earth (axis=-z)'
        
        #plot trajectory and highlighted points if requested
        plt.plot(x,y,'b-')
        if ptflag:
            plt.plot(xpts,ypts,'go')
        plt.gca().set_aspect('equal','datalim')
        plt.show()
    
    #Maintenence functions: lots of conditionals for setting and interpreting flags
    #If this is opaque, that's 100% okay as long as you understand what 'axis' and 'facing'
    #determine in plot_orbit()
    def set_flags(self,axis,facing):
        if (facing%2) == 0:
            if axis[-1] == 'x':
                xflag=2
                yflag=3
            elif axis[-1] == 'y':
                xflag=-1
                yflag=3
            elif axis[-1] == 'z':
                xflag=1
                yflag=2
            else:
                print 'invalid axis'
                return
            if axis[0]=='-':
                xflag*=-1
        else:
            if axis[-1] == 'x':
                xflag=3
                yflag=-2
            elif axis[-1] == 'y':
                xflag=3
                yflag=1
            elif axis[-1] == 'z':
                xflag=2
                yflag=-1
            else:
                print 'invalid axis'
                return
            if axis[0]=='-':
                yflag*=-1
        if facing >= 2:
            xflag*=-1
            yflag*=-1
        return xflag,yflag
    def use_flags(self,data,xflag,yflag):
        if xflag < 0:
            x = -data[-xflag-1]
        else:
            x = data[xflag-1]
        if yflag < 0:
            y = -data[-yflag-1]
        else:
            y = data[yflag-1]
        return x,y
    #useful observation extractor
    def get_obs(self,filename,distance):
        xobs=(np.genfromtxt(filename,skip_header=1,delimiter=',',usecols=11,unpack=True))/1000.
        yobs=(np.genfromtxt(filename,skip_header=1,delimiter=',',usecols=13,unpack=True))/1000.
        xobs=xobs[yobs==yobs]
        yobs=yobs[yobs==yobs]
        xobs*=distance
        yobs*=distance
        return np.array([xobs,yobs])
    def get_err(self,filename,distance): #error extractor --- UNIMPLEMENTED
        pass
    def get_eps(self,filename,distance): #epoch extractor --- UNIMPLEMENTED
        pass
    def units(self,inputs,in_units='solar',out_units='SI'):
        pass
