import numpy as np

for i in range(1):
    with np.load('F:/PSF_modeling/acswfc1/narrow_gauss/acs_wfc100_psf.fits'+str(i)+'.npz') as d:
        guess = d['guess']
        px = d['px']
        py = d['py']
        plh= d['plh']
        plv =d['plv']
        pph = d['pph']
        ppv = d['ppv']
        pch = d['pch']
        pcv = d['pcv']
        ints = d['ints']
        good_hs = d['good_hs']
        good_vs = d['good_vs']
        good_horiz = d['good_horiz']
        good_vert = d['good_vert']
        fname = d['fname']
    np.savez_compressed('C:/Users/Dominic/Documents/astro/Research/dev_code/image_fitting/past_original_data/samples/narrow_gauss'+str(i),
        guess=guess,px=px,py=py,plh=plh,plv=plv,pph=pph,ppv=ppv,pch=pch,pcv=pcv,ints=ints,good_hs=good_hs,good_vs=good_vs,
        good_horiz=good_horiz,good_vert=good_vert,fname=fname)
for i in range(4):
    with np.load('F:/PSF_modeling/acswfc1/test/acs_wfc100_psf.fits'+str(i)+'.npz') as d:
        guess = d['guess']
        px = d['px']
        py = d['py']
        plh= d['plh']
        plv =d['plv']
        pph = d['pph']
        ppv = d['ppv']
        pch = d['pch']
        pcv = d['pcv']
        ints = d['ints']
        good_hs = d['good_hs']
        good_vs = d['good_vs']
        good_horiz = d['good_horiz']
        good_vert = d['good_vert']
        fname = d['fname']
    np.savez_compressed('C:/Users/Dominic/Documents/astro/Research/dev_code/image_fitting/past_original_data/samples/test'+str(i),
        guess=guess,px=px,py=py,plh=plh,plv=plv,pph=pph,ppv=ppv,pch=pch,pcv=pcv,ints=ints,good_hs=good_hs,good_vs=good_vs,
        good_horiz=good_horiz,good_vert=good_vert,fname=fname)
        
with np.load('F:/PSF_modeling/acswfc1/narrow_gauss/acs_wfc100_psf.fits.npz') as d:
    guess = d['guess']
    ints = d['ints']
    img = d['img']
    brightest=d['brightest']
    lows=d['lows']
    highs=d['highs']
    fname = d['fname']
np.savez_compressed('C:/Users/Dominic/Documents/astro/Research/dev_code/image_fitting/past_original_data/samples/narrow_gauss',
    guess=guess,ints=ints,img=img,brightest=brightest,lows=lows,highs=highs,fname=fname)