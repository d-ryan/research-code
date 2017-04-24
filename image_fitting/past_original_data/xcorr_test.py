for xshift, yshift:
    shifted_tinytim_psf = generate_tinytim_psf(xshift, yshift)
    xcorr_shift = np.dot(shifted_tinytim_psf, data)/np.sqrt(np.dot(shifted_tinytim_psf, shifted_tinytim_psf) * np.dot(data, data))
# do for a combination of xshift,yshift and find one that maximizes xcorr_shift
# all matrix dot products are b/w 2 1-d vectors so they yield a scalar
# data, shifted_tinytim_psf could start out as 2-d arrays, but jason indexes
#     them with np.where() to grab relevant pixels as 1-d array