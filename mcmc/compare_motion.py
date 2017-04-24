import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = 'fomalhaut_new.csv'

def cv(t,x0,v0):
    return x0 + v0*t

def ca(t,x0,v0,a):
    return x0 + v0*t + 0.5*a*(t**2.)


t=np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9), unpack=True).T.ravel()

x = np.genfromtxt(filename, skip_header=1,delimiter=',',usecols=(11),unpack=True).T.ravel()/1000.
y = np.genfromtxt(filename, skip_header=1,delimiter=',',usecols=(13),unpack=True).T.ravel()/1000.

sig_x = np.genfromtxt(filename, skip_header=1,delimiter=',',usecols=(12),unpack=True).T.ravel()/1000.
sig_y = np.genfromtxt(filename, skip_header=1,delimiter=',',usecols=(14),unpack=True).T.ravel()/1000.

pvx,cvx = curve_fit(cv,t,x,sigma=sig_x,absolute_sigma = True)
pvy,cvy = curve_fit(cv,t,y,sigma=sig_y,absolute_sigma = True)

pax,cax = curve_fit(ca,t,x,sigma=sig_x,absolute_sigma = True)
pay,cay = curve_fit(ca,t,y,sigma=sig_y,absolute_sigma = True)

start = np.min(t)
times = np.linspace(start-365.2425*30,start+365.2425*30,2000)

print 'Optimal values, unaccelerated'
print 'x: ','x0 = ',pvx[0],' v0x = ',pvx[1]
print 'y: ','y0 = ',pvy[0],' v0y = ',pvy[1]
print 'Accelerated'
print 'x: ','x0 = ',pax[0],' v0x = ',pax[1],' ax = ',pax[2]
print 'y: ','y0 = ',pay[0],' v0y = ',pay[1],' ay = ',pay[2]

plt.errorbar(t,x,yerr=sig_x,fmt='ro',ecolor='k',elinewidth=2,capthick=2,ms=2)
plt.errorbar(t,y,yerr=sig_y,fmt='bo',ecolor='k',elinewidth=2,capthick=2,ms=2)

plt.plot(t,x,'ro',t,y,'bo')
plt.plot(times,cv(times,*pvx),'r-')
plt.plot(times,ca(times,*pax),'r--')
plt.plot(times,cv(times,*pvy),'b-')
plt.plot(times,ca(times,*pay),'b--')
plt.show()
