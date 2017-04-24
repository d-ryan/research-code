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
from spikefit_wfpc_dev import get_data,padding,derotate,line,parab,cubic

prefix = 'F:/BetaPic/wfpc2/'

with np.load('F:/BetaPic/wfpc2/info.npz') as d:
    snames=list(d['snames'])

all_ints = np.zeros((len(snames),2))
fnames = []
angles = []
pads = []
bigshapes = []
shapes = []
for i,s in enumerate(snames):
    with np.load(s+'.npz') as d:
        ints=d['ints']
        fnames.append(str(d['fname']))
        angles.append(d['angle'])
        pads.append(d['pad'])
        bigshapes.append(len(d['px'][1,:]))
        
    all_ints[i,:]=ints[2,:]

for i,f in enumerate(fnames):
    img,drzangle = get_data(prefix+f)
    shapes.append(img.shape[0])
    
with open('wfpc2_ints.csv','wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['File','x (cubic)','y (cubic)','Derotate Angle','Padding [pix]','Big','Shape'])
    for i in range(len(fnames)):
        writer.writerow([fnames[i],all_ints[i,0],all_ints[i,1],np.degrees(angles[i]),pads[i],bigshapes[i],shapes[i]])