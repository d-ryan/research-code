import numpy as np

#simple, stupid n-body code.
#Likely slow enough to be useless.
#Could possibly be improved with scipy solvers. 

GSI = 6.674e-11
Msun = 1.988e30
AU = 1.496e11
year = 3.1556952e7
Gred = 4.*np.pi**2.

class solver(object):
    rtol=1e-6
    def __init__(self,gbodies):
        self.s_ind=0
        self.t=0.
        self.h=0.01
        self.n = np.size(gbodies)
        self.gbodies = gbodies
        self.forces = np.zeros((n,n))
        self.nets = np.zeros((n))
        self.tempks = np.zeros((n,6))
    def step(self):
        
        
        
        
        
        if self.s_ind%10==0:
            self.save()
        if self.s_ind%100==0:
            self.set_timestep()
        self.s_ind+=1

    def f(self,dt=0.,k=None):
        #Grab y
        
        #Add appropriate correction to y
        if k is not None:
            y+=dt*k
    
    def fill_matrix(self):
        for i,body_i in enumerate(self.gbodies):
            for j, body_j in enumerate(self.gbodies[i+1:]):
                self.forces[i,j] = body_i.forcefrom(body_j)
                self.forces[j,i] = -self.forces[i,j]
            self.nets[i]=np.sum(self.forces[i,:])
            
        
    def rk4(self,h):
        k1 = self.f()
        k2 = self.f(dt=h/2.,k=k1)
        k3 = self.f(dt=h/2.,k=k2)
        k4 = self.f(dt=h,k=k3)
        return h/6.*(k1+2*k2+2*k3+k4)
        
    def set_timestep(self):

def f(x,y,z,vx,vy,vz):
    
    
    return vx,vy,vz,ax,ay,az

class gbody(object):
    id = 0
    def __init__(self,name,m,x,y,z,vx,vy,vz):
        self.name = name
        self.m=m
        store_y(self,x,y,z,vx,vy,vz)
        self.old=[x,y,z,vx,vy,vz]
        self.id = gbody.id
        gbody.id+=1
    
    def accfrom(self,body):
        num = Gred*body.m*self.relpos(body)
        denom = self.dist(body)**3
        return num/denom
        
    def netaccfrom(self,bodies):
        a=0.
        for body in bodies:
            a+=accfrom(body)
        return a
    
    def netforcefrom(self,bodies):
        return self.m*self.netaccfrom(bodies)
        
    def accon(self,body):
        return -self.accfrom(body)
    
    def forcefrom(self,body):
        return self.m*self.accfrom(body)
    
    def relpos(self,body):
        return body.vecr-self.vecr
    
    def dist(self,body):
        return np.sqrt(np.sum(self.relpos(body)**2.))
        
    def relv(self,body):
        return body.vecv-self.vecv
        
    def relspd(self,body):
        return np.sqrt(np.sum(self.relv(body)**2.))
    
    def forceon(self,body):
        return -self.forcefrom(body)
    
    def update_y(self,x,y,z,vx,vy,vz):
        self.old = [self.x,self.y,self.z,self.vx,self.vy,self.vz]
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz

    def store_y(self,x,y,z,vx,vy,vz):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        
    def reset_y(self):
        self.x = self.old[0]
        self.y = self.old[1]
        self.z = self.old[2]
        self.vx = self.old[3]
        self.vy = self.old[4]
        self.vz = self.old[5]
    
    @property
    def vecr(self):
        return np.array([self.x,self.y,self.z])
    
    @property
    def r(self):
        return np.sqrt(np.sum(self.vecr**2.))
    
    @property
    def vecv(self):
        return np.array([self.vx,self.vy,self.vz])
        
    @property
    def v(self):
        return np.sqrt(np.sum(self.vecv**2.))

jupiter = gbody('jupiter',1./1047.,5.202,0.,0.,0.,2.*np.pi*(5.202**(-0.5)),0.)
io
europa
ganymede
callisto
sun = gbody('sun',1.,-jupiter.x/1047.,0.,0.,0.,-jupiter.vy/1047.,0.)