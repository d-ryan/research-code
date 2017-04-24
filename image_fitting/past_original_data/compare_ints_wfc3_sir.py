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

prefix = 'F:/stis_go13362/'
#prefix = 'F:/sirius_astrometry/'

with np.load(prefix+'info.npz') as d:
    snames=list(d['snames'])

all_ints = np.zeros((len(snames),3,2))
fnames = []
angles = []
pads = []
bigshapes = []
shapes = []
diffs = []
for i,s in enumerate(snames):
    with np.load(s+'.npz') as d:
        ints=d['ints']
        fnames.append(str(d['fname']))
        angles.append(d['angle'])
        pads.append(d['pad'])
        bigshapes.append(len(d['px'][1,:]))
        diffs.append(d['diff'])
        
    all_ints[i,:,:]=ints[:,:]

for i,f in enumerate(fnames):
    img,drzangle = get_data(prefix+f,index=0,drz=False)
    #img,drzangle = get_data(prefix+f,index=1,drz=False)
    shapes.append(img.shape[0])
    
with open(prefix+'testfmh.csv','wb') as csvfile:
    writer = csv.writer(csvfile)
    row = ['File + method','x','y','Derotate Angle','Padding [pix]','Big','Shape','Diff']
    writer.writerow(row)
    j=0
    nums = []
    for i,f in enumerate(fnames):
        j+=1
        stri = str(3*i+2)
        A = fnames[i]+' lin' #Filename + method
        B = all_ints[i,0,0] #x
        C = all_ints[i,0,1] #y
        D = np.degrees(angles[i]) #angle
        E = pads[i] #pad
        F = bigshapes[i] #bigshp
        G = shapes[i] #shp
        H = diffs[i] #diff - last non-formula
        I = '=F'+stri+'-G'+stri #big-shp - first formula
        J = '=E'+stri
        K = '=I'+stri+'-E'+stri
        L = '=B'+stri+'-E'+stri
        M = '=C'+stri+'-E'+stri
        N = '=FLOOR(F'+stri+'/2,1)'
        O = '=(B'+stri+'-N'+stri+')*COS(RADIANS(D'+stri+'))+(C'+stri+'-N'+stri+')*SIN(RADIANS(D'+stri+'))+N'+stri+'-E'+stri
        P = '=(-1)*(B'+stri+'-N'+stri+')*SIN(RADIANS(D'+stri+'))+(C'+stri+'-N'+stri+')*COS(RADIANS(D'+stri+'))+N'+stri+'-E'+stri
        Q = ''
        R = j
        S = '=OFFSET(O$2,3*(ROW()-2),0)'
        T = '=OFFSET(P$2,3*(ROW()-2),0)'
        U = '=OFFSET(O$2,3*(ROW()-2)+1,0)'
        V = '=OFFSET(P$2,3*(ROW()-2)+1,0)'
        W = '=OFFSET(O$2,3*(ROW()-2)+2,0)'
        X = '=OFFSET(P$2,3*(ROW()-2)+2,0)'
        row = [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X]
        writer.writerow(row)
        stri = str(3*i+3)
        j+=1
        A = fnames[i]+' qdr'
        B = all_ints[i,1,0]
        C = all_ints[i,1,1]
        I = '=F'+stri+'-G'+stri #big-shp - first formula
        J = '=E'+stri
        K = '=I'+stri+'-E'+stri
        L = '=B'+stri+'-E'+stri
        M = '=C'+stri+'-E'+stri
        N = '=FLOOR(F'+stri+'/2,1)'
        O = '=(B'+stri+'-N'+stri+')*COS(RADIANS(D'+stri+'))+(C'+stri+'-N'+stri+')*SIN(RADIANS(D'+stri+'))+N'+stri+'-E'+stri
        P = '=(-1)*(B'+stri+'-N'+stri+')*SIN(RADIANS(D'+stri+'))+(C'+stri+'-N'+stri+')*COS(RADIANS(D'+stri+'))+N'+stri+'-E'+stri
        R = j
        row = [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X]
        writer.writerow(row)
        stri = str(3*i+4)
        j+=1
        A = fnames[i]+' cub'
        B = all_ints[i,2,0]
        C = all_ints[i,2,1]
        I = '=F'+stri+'-G'+stri #big-shp - first formula
        J = '=E'+stri
        K = '=I'+stri+'-E'+stri
        L = '=B'+stri+'-E'+stri
        M = '=C'+stri+'-E'+stri
        N = '=FLOOR(F'+stri+'/2,1)'
        O = '=(B'+stri+'-N'+stri+')*COS(RADIANS(D'+stri+'))+(C'+stri+'-N'+stri+')*SIN(RADIANS(D'+stri+'))+N'+stri+'-E'+stri
        P = '=(-1)*(B'+stri+'-N'+stri+')*SIN(RADIANS(D'+stri+'))+(C'+stri+'-N'+stri+')*COS(RADIANS(D'+stri+'))+N'+stri+'-E'+stri
        R = j
        row = [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X]
        writer.writerow(row)
        