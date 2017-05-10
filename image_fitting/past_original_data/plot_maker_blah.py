import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits

prefix = 'F:/PSF_modeling/acswfc1/'

xstds = []
xmeans = []
ystds = []
ymeans = []
i = 0
i_s = []
x_coords = []
y_coords = []
save_ints = np.zeros((2,6,42))

for filename in os.listdir(prefix):
    if filename[-4:]=='fits':
        hdulist = fits.open(prefix+filename)
        img = hdulist[0].data
        x_coord = hdulist[0].header['X_PSF']
        y_coord = hdulist[0].header['Y_PSF']
        hdulist.close()
        
        savename1 = prefix+'narrow_gauss/'+filename+'0.npz'

        with np.load(savename1) as d:
            px = d['px']
            py = d['py']
            cx = d['cx']
            cy = d['cy']
            good_horiz = d['good_horiz']
            good_vert = d['good_vert']
            good_hs = d['good_hs']
            good_vs = d['good_vs']

        if i==0:
            np.savez_compressed('C:/Users/Dominic/Documents/astro/Research/dev_code/image_fitting/past_original_data/samples/plt_ex',
                img=img,px=px,py=py,cx=cx,cy=cy,good_horiz=good_horiz,good_vert=good_vert,good_hs=good_hs,good_vs=good_vs)

        savename2 = prefix+'narrow_gauss/'+filename+'.npz'
        with np.load(savename2) as d:
            ints = d['ints']
            

        xmean = np.mean(good_horiz-1174)
        xmeans.append(xmean)
        xstd = np.std(good_horiz)
        xstds.append(xstd)
        ymeans.append(np.mean(good_vert-1174))
        ystds.append(np.std(good_vert))
        x_coords.append(x_coord)
        y_coords.append(y_coord)
        i_s.append(i)
        save_ints[:,:,i]=ints[0,:,:]
        i+=1
    
prefix = 'F:/PSF_modeling/acswfc2/'
for filename in os.listdir(prefix):
    if filename[-4:]=='fits':
        hdulist = fits.open(prefix+filename)
        img = hdulist[0].data
        x_coord = hdulist[0].header['X_PSF']
        y_coord = hdulist[0].header['Y_PSF'] + 2048
        
        savename1 = prefix+'narrow_gauss/'+filename+'0.npz'
        
        with np.load(savename1) as d:
            px = d['px']
            py = d['py']
            cx = d['cx']
            cy = d['cy']
            good_horiz = d['good_horiz']
            good_vert = d['good_vert']
            good_hs = d['good_hs']
            good_vs = d['good_vs']
            
        savename2 = prefix+'narrow_gauss/'+filename+'.npz'
        with np.load(savename2) as d:
            ints = d['ints']
        
        xmean = np.mean(good_horiz-1174)
        xmeans.append(xmean)
        xstd = np.std(good_horiz)
        xstds.append(xstd)
        ymeans.append(np.mean(good_vert-1174))
        ystds.append(np.std(good_vert))
        x_coords.append(x_coord)
        y_coords.append(y_coord)
        i_s.append(i)
        save_ints[:,:,i]=ints[0,:,:]
        i+=1
    
np.savez_compressed('C:/Users/Dominic/Documents/astro/Research/dev_code/image_fitting/past_original_data/samples/ints_stats',
    xmeans=xmeans,ymeans=ymeans,xstds=xstds,ystds=ystds,x_coords=x_coords,y_coords=y_coords,i_s=i_s,ints=save_ints)
