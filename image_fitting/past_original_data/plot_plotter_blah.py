import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits

with np.load('samples/plt_ex.npz') as d:
    img=d['img']
    px=d['px']
    py = d['py']
    cx = d['cx']
    cy = d['cy']
    good_horiz = d['good_horiz']
    good_vert = d['good_vert']
    good_hs = d['good_hs']
    good_vs = d['good_vs']

fig = plt.figure()
ax1=fig.add_subplot(211)
plt.errorbar(good_hs,good_horiz-1174,yerr=cx[1,1,good_hs.astype('int')],fmt='o',label='horizontal')
plt.errorbar(good_vs,good_vert-1174,yerr=cy[1,1,good_vs.astype('int')],fmt='o',label='vertical')
plt.xlabel('image slice [pix]')
plt.ylabel('raw centering error [pix]')
plt.legend()
print np.mean(good_horiz-1174)
print np.std(good_horiz)
ax2=fig.add_subplot(223)
plt.xlabel('x error histogram [pix]')
plt.hist(good_horiz-1174,bins=20)
ax3=fig.add_subplot(224)
plt.xlabel('y error histogram [pix]')
plt.hist(good_vert-1174,bins=20)
plt.hist(good_vert-1174,bins=20)
plt.show()

with np.load('samples/ints_stats.npz') as d:
    x_coords = d['x_coords']
pdb.set_trace()