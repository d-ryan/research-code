import numpy as np
import numpy as np
from astropy.io import fits
from skimage.transform._warps_cy import _warp_fast
from scipy.optimize import curve_fit,root
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
import csv
from spikefit_gen_dev import get_data,padding,derotate,line,parab,cubic

prefix = 'F:/PSF_modeling/acswfc2/test/'

with np.load(prefix+'info.npz') as d:
    snames=list(d['snames'])

dsox=[]
dsoy=[]
dsrx=[]
dsry = []
gx = []
gy = []
sx = []
sy =[]
px = []
py = []
bx = []
by = []
z = []

for i,s in enumerate(snames):
    with np.load(s+'.npz') as d:
        ints=d['ints']
        lows=d['lows']
        highs=d['highs']
        brightest=d['brightest']
    dsox.append(ints[0,0,0])
    dsoy.append(ints[0,1,0])
    dsrx.append(ints[0,0,1])
    dsry.append(ints[0,1,1])
    gx.append(ints[0,0,2])
    gy.append(ints[0,1,2])
    sx.append(ints[0,0,2])
    sy.append(ints[0,1,2])
    bx.append(ints[0,0,4])
    by.append(ints[0,1,4])
    px.append(ints[0,0,5])
    py.append(ints[0,0,5])
    z.append(i)
    
plt.plot(1174,1174,'bo')
plt.plot(dsox,dsoy,'ro',label='Double slit offset')
plt.plot(px,py,'go',label='Bond Parabola')
plt.plot(bx,by,'yo',label='Brightest Pixel Trace')
plt.plot(sx,sy,'co',label='sinc^2')
plt.legend()
plt.title('Bias in spikefit techniques [IN PROGRESS]')
plt.xlabel('x [pix]')
plt.ylabel('y [pix]')
plt.gca().set_aspect('equal','datalim')

plt.savefig(prefix+'full_comp.pdf')
plt.savefig(prefix+'full_comp.png')

plt.show()

#From reflex2.py
#Figure out how to make this work for keying the location on the chip
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