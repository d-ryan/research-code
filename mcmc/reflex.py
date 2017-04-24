import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from orbitclass import orbit
from scipy.interpolate import interp1d
import jdcal
import pdb
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.markers import MarkerStyle
from datetime import date


'''
Reflex motion stuff:
--need an orbit class for each planet
--for a smooth range of times, calculate planet positions
--calculate star positions at those times FROM planet positions
--apply parallax/proper motion if need be
--plot'''

def plot_colormap(x,y,z,lcm='brg',pcm=None,line=True,pts=True,linewidth=None,axes=False,
    xlabel=None,ylabel=None,title=None,filename=None):
    '''Plot (x,y) data with a color gradient according to z, as points or lines.'''
    fig,ax = plt.subplots()
    norm=plt.Normalize(z.min(),z.max())
    if line:
        points = np.array([x,y]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = LineCollection(segments,cmap=lcm,norm=norm)
        lc.set_array(z)
        if linewidth is not None:
            lc.set_linewidth(linewidth)
        ax.add_collection(lc)
    if pts:
        if pcm is not None:
            sc=ax.scatter(x,y,c=z,cmap=pcm,norm=norm, linewidth=0)
        else:
            sc=ax.scatter(x,y,c=z,cmap=lcm,norm=norm, linewidth=0)
    plt.xlim(x.min()-0.1*(x.max()-x.min()),x.max()+0.1*(x.max()-x.min()))
    plt.ylim(y.min()-0.1*(y.max()-y.min()),y.max()+0.1*(y.max()-y.min()))
    ax.set_aspect('equal','datalim')
    if axes:
        plt.axhline(color='k')
        plt.axvline(color='k')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if line:
        cb=plt.colorbar(lc)
    if pts and not line:
        cb=plt.colorbar(sc)
    today=jdcal.gcal2jd(date.today().year,date.today().month,date.today().day)[1]
    cb.set_label('MJD (Today: '+str(today)+')')
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def get_orbits(filename):
    '''
    Return list of orbital parameters read in from file specified by *filename*.
    *filename* should be a csv file in the same directory, or the full path to
    such a csv file. The csv file should contain a list of planets and their orbital
    parameters for a given system, in the order: a, e, i, argument of periastron,
    longitude of ascending node, epoch of periastron, planet mass, star mass,
    and system distance from earth.
    
    It is possible to input differing star masses for each planet. Doing so gives
    meaningless garbage results. Check to ensure the star mass and system distance
    are the same for each planet.
    '''
    data=np.genfromtxt(filename,delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7,8,9))
    f=open(filename,'r')
    f.readline()
    orbitlist=[]
    if len(data.shape)==1:
        orbitlist.append(orbit(a=data[0],e=data[1],i=data[2],argp=data[3],lan=data[4],tau=data[5],m_planet=data[6],m=data[7],dist=data[8]))
        id=f.readline().split(',')[0]
        orbitlist[0].set_id(id)
    else:
        rows=data.shape[0]
        for i in range(rows):
            orbitlist.append(orbit(a=data[i,0],e=data[i,1],i=data[i,2],argp=data[i,3],lan=data[i,4],tau=data[i,5],m_planet=data[i,6],m=data[i,7],dist=data[i,8]))
            id=f.readline().split(',')[0]
            orbitlist[i].set_id(id)
    f.close()
    return orbitlist

def make_times(start,end,number):
    '''
    Return *number* evenly spaced epochs for for plotting a reflex motion curve.
    If *start* is an earlier epoch than *end*, epochs will be in ascending order.
    If *start* is a later epoch than *end*, epochs will be in descending order--
    that is, having *start*>*end* means plotting backward in time.
    '''
    if start>end:
        times=np.linspace(end,start,number)[::-1]
    else:
        times=np.linspace(start,end,number)
    return times
    
def ellipse_coeff(x,y):
    '''
    Algebraically solve for the coefficients in the equation:
    Ax^2 + Bxy + Cy^2 + Dx + Ey + 1 = 0
    given five (x,y) data points.
    
    Only useful in the case of a one-planet system. Solves for the
    exact geometry of the orbital ellipse as projected onto the
    plane of the sky. This allows the maximum astrometric amplitude
    to be determined (in the case of Beta Pic b with a near edge-on
    orbit, this is notably larger than the amplitude in either x or y.
    However, a face-on, circular orbit doesn't HAVE a long axis on which
    the projected motion appears greatest.)
    '''
    if (len(x) != 5) or (len(y) != 5):
        print 'x,y arrays must have length 5'
        return
    x=np.array(x)
    y=np.array(y)
    M=np.column_stack((x**2,x*y,y**2,x,y))
    Minv=np.linalg.inv(M)
    vec=np.dot(Minv,-np.ones(5))
    A=vec[0]
    B=vec[1]
    C=vec[2]
    D=vec[3]
    E=vec[4]
    yc=(2.*A*E-B*D)/(B**2.-4.*C*A)
    xc=(D+B*yc)/(-2.*A)
    theta = np.arctan2(B,A-C)*0.5
    a = np.sqrt(2.*(A*xc**2.+B*xc*yc+C*yc**2.-1)/(A+C+B/np.sin(2.*theta)))
    b = a*np.sqrt((A+C+B/np.sin(2.*theta))/(A+C-B/np.sin(2.*theta)))
    return a,b,theta,xc,yc
    
def time_plots(times,orbitlist):
    star_xy=mk_star_xy(times,orbitlist)
    star_net=np.sum(star_xy,axis=1)
    xplot=star_net[0,:]/orbitlist[0].dist*1000.
    yplot=star_net[1,:]/orbitlist[0].dist*1000.
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    plt.xlabel('MJD')
    plt.ylabel('mas')
    plt.title('Beta Pic major axis reflex motion')

    if len(orbitlist)==1:
        vals=np.linspace(0.,8.*np.pi/5.,5.)
        orb=orbitlist[0]
        a,b,theta,xc,yc=ellipse_coeff(orb.x(vals)*orb.m_planet/orb.m,orb.y(vals)*(-orb.m_planet/orb.m))
        along_a=(star_net[0,:]-xc)*np.cos(theta)+(star_net[1,:]-yc)*np.sin(theta)
        along_a*=1000./orbitlist[0].dist
        ax1.plot(times,along_a)
        plt.ylim(1.5*np.min(along_a),1.5*np.max(along_a))
    print times[np.argmax(along_a)]
    print times[np.argmin(along_a)]
    interp=interp1d(times,along_a)
    
    dates=np.genfromtxt('MAST_beta_pictoris.csv',skiprows=5,delimiter=',',usecols=(24,),skip_header=1,unpack=True)
    dates = dates[dates==dates]
    uniques = np.unique(dates)
    uniques=uniques[1:]
    plt.plot(uniques,interp(uniques),'bo')
    
    
    fig1.savefig('testplotall.png')

def mk_star_xy(times,orbitlist):
    star_xy=np.zeros((2,len(orbitlist),len(times)))
    for i,orb in enumerate(orbitlist):
        data=orb.cartesian(orb.time2true(times))
        star_xy[0,i,:]=data[0]*(orb.m_planet/orb.m)
        star_xy[1,i,:]=data[1]*(-orb.m_planet/orb.m)
    return star_xy

def xy_plot(times,orbitlist):
    star_xy=mk_star_xy(times,orbitlist)
    star_net=np.sum(star_xy,axis=1)

    xplot=star_net[0,:]/orbitlist[0].dist*1000.
    yplot=star_net[1,:]/orbitlist[0].dist*1000.

    xplot-=xplot[0]
    yplot-=yplot[0]
    plot_colormap(xplot,yplot,times,lcm='brg',filename='testtesttest.png',xlabel='milliarcseconds',ylabel='milliarcseconds',title='HR8799 simulated reflex motion trace',linewidth=2)

if __name__=='__main__':
    #plots x,y,aproj v t
    filename = 'beta_pic_test.csv'
    start=jdcal.gcal2jd(date.today().year,date.today().month,date.today().day)[1]
    n_years=30
    times=make_times(start+365.2425*n_years,start-365.2425*n_years,(n_years*24.)+1.)
    time_plots(times,get_orbits(filename))



    #Plots xvy
    filename='hr8799_params.csv'
    start=jdcal.gcal2jd(date.today().year,date.today().month,date.today().day)[1]
    n_years=150
    times=make_times(start,start-365.2425*n_years,n_years+1)
    xy_plot(times,get_orbits(filename))