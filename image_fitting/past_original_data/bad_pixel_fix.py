import numpy
import numpy as np

def mask_bad_pix(im, inst=None, Nsig=7, thr_min=0.25):
    """
    Remove bad pixels by comparing their value to mean of neighbor pixels.
    
    Inputs:
        im= image array to mark bad pixels in.
        inst= str; name of instrument im came from; only needed if instrument
                has known bad pixel locations.
        Nsig= # of standard deviations above the mean required to mark pixel as bad.
        thr_min= hard minimum value above the mean required to mark pixel as bad.
    
    Output:
        numpy array with bad pixels = NaN.
    """
    
    print "\nMarking bad pixels (may take a few minutes)..."
    
    mat_list = []
    
    # Create zero arrays same size as im but padded by 2 pixels on each side.
    # Shift im around edge of an 8-pixel square, with starting point
    # (0,0), to make 8 new arrays with differentially-shifted im.
    for yx in [(2,2), (1,2), (0,2), (0,1), (0,0), (1,0), (2,0), (2,1)]:
        mat = numpy.zeros((im.shape[0]+4, im.shape[1]+4))
        mat[yx[0]:yx[0]+im.shape[0], yx[1]:yx[1]+im.shape[1]] = im.copy()
        mat_list.append(mat)
    
    # Calculate the mean of nearest neighbors (ignore NaN).
    nebMean_mat = np.nanmean(mat_list, axis=0)
    # Calculate standard deviation of nearest neighbors (ignore NaN).
    # (Note: equivalent to deprecated scipy.stats.nanstd(mat_list, axis=0, bias=True))
    nebStd_mat = np.nanstd(mat_list, axis=0)
    
    # Trim padding off of nebMean_mat and nebStd_mat so they match shape of im.
    # NOTE: intentionally starting at index 1, based on shifting above.
    nebMeanArr = nebMean_mat[1:im.shape[0]+1, 1:im.shape[1]+1]
    nebStdArr = nebStd_mat[1:im.shape[0]+1, 1:im.shape[1]+1]
    # Set threshold requirement for a "bad" pixel as Nsig standard deviations
    # above the mean or thr_min, whichever is smaller.
    thresh = Nsig*nebStdArr.copy()
    thresh[thresh > thr_min] = thr_min
    # Find all pixels in im that exceed thresh relative to neighbors.
    diff = numpy.abs(nebMeanArr - im) - thresh
    whb = numpy.where(diff > 0.)
    
    # Replace all "bad" pixels with NaN.
    im[whb] = numpy.nan
    
    # Mask known bad pixels according to instrument specified.
    if inst=="NickelDIC":
        im[:, (256, 783, 784)] = numpy.nan
    
    print "%d bad pixels (%.1f%%) marked as NaN (not counting known bad pixels)" % (whb[0].shape[0], 100*whb[0].shape[0]/float(im.size))
    
    return im


def fill_bad_pix(im, numNeb=1):
    """
    Replace all NaN pixels with the median of their neighbors.
    numNeb= number of neighbors to use, counting away from bad pixel in 1-D.
            E.g., numNeb=1 gives 8 neighbors in 3x3 patch centered on bad pixel.
    """
    # Get indices of all NaN pixels in im.
    wh_nan = numpy.where(numpy.isnan(im))
    # Make copy of im for iteration purposes.
    iter_im = im.copy()
    # Find last rows & columns of array can index without problems (see below).
    im_edge_y = im.shape[0] - numNeb
    im_edge_x = im.shape[1] - numNeb
    
    # Get median of entire im array (excluding NaN).
    im_med = np.nanmedian(im.flatten())
    
    # Iterate over all NaN pixels to replace them one by one.
    # NOTE: this type of 'for' loop can be very slow, but it is simple to use.
    for ii in range(wh_nan[0].shape[0]):
        iy, ix = wh_nan[0][ii], wh_nan[1][ii]
        # Around edges of array, just fill NaN with overall image median.
        # Simple way to avoid indexing problems when going off edge of array.
        if (iy < numNeb) | (ix < numNeb) | (iy >= im_edge_y) | (ix >= im_edge_x):
            im[iy,ix] = im_med
        # For rest of NaN, replace with median of surrounding 5x5 pixel patch.
        else:
            # Get 5x5 pixel patch around NaN in original im array.
            patch = iter_im[iy-numNeb:iy+numNeb+1, ix-numNeb:ix+numNeb+1]
            # Replace NaN with median of patch (excluding any NaN in patch).
            im[iy,ix] = np.nanmedian(patch.flatten())
    
    # Fill any remaining stubborn NaN with median of entire im array.
    im[numpy.where(numpy.isnan(im))] = im_med
    
    return im