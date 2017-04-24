#Radon redraft
#Plan: Automatic iteration to the desired
#  decimal depth.
#Do not interpolate the cost function

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.ndimage.interpolation import map_coordinates

def samplingRegion(size_window, theta = [45, 135], m = 0.2, M = 0.8, step = 1, decimals = 2):
    """This function returns all the coordinates of the sampling region, the center of the region is (0,0)
    When applying to matrices, don't forget to SHIFT THE CENTER!
    Input:
        size_window: the radius of the sampling region. The whole region should thus have a length of 2*size_window+1.
        theta: the angle range of the sampling region, default: [45, 135] for the anti-diagonal and diagonal directions.
        m: the minimum fraction of size_window, default: 0.2 (i.e., 20%). In this way, the saturated region can be excluded.
        M: the maximum fraction of size_window, default: 0.8 (i.e., 80%). Just in case if there's some star along the diagonals.
        step: the seperation between sampling dots (units: pixel), default value is 1pix.
        decimals: the precisoin of the sampling dots (units: pixel), default value is 0.01pix.
    Output: (xs, ys)
        xs: x indecies, flattend.
        ys: y indecies, flattend.
    Example:
        If you call "xs, ys = samplingRegion(5)", you will get:
        xs: array([-2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83,  2.83, 2.12,  1.41,  0.71, -0.71, -1.41, -2.12, -2.83]
        ys: array([-2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83, -2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83]))
    """
    theta = np.array(theta)
    zeroDegXs = np.append(np.arange(-int(size_window*M), -int(size_window*m) + 0.1 * step, step), np.arange(int(size_window*m), int(size_window*M) + 0.1 * step, step))
    #create the x indecies for theta = 0, will be used in the loop.
    zeroDegYs = np.zeros(zeroDegXs.size)
    
    xs = np.zeros((np.size(theta), np.size(zeroDegXs)))
    ys = np.zeros((np.size(theta), np.size(zeroDegXs)))
    
    for i, angle in enumerate(theta):
        degRad = np.deg2rad(angle)
        angleDegXs = np.round(zeroDegXs * np.cos(degRad), decimals = decimals)
        angleDegYs = np.round(zeroDegXs * np.sin(degRad), decimals = decimals)
        xs[i, ] = angleDegXs
        ys[i, ] = angleDegYs
    
    xs = xs.flatten()
    ys = ys.flatten()

    return xs, ys


def smoothCostFunction(costFunction, halfWidth = 0):
    """
    smoothCostFunction will smooth the function within +/- halfWidth, i.e., to replace the value with the average within +/- halfWidth pixel.
    Input:
        costFunction: original cost function, a matrix.
        halfWdith: the half width of the smoothing region, default = 0 pix.
    Output:
        newFunction: smoothed cost function.
    """
    if halfWidth == 0:
        return costFunction
    else:
        newFunction = np.zeros(costFunction.shape)
        rowRange = np.arange(costFunction.shape[0], dtype=int)
        colRange = np.arange(costFunction.shape[1], dtype=int)
        rangeShift = np.arange(-halfWidth, halfWidth + 0.1, dtype=int)
        for i in rowRange:
            for j in colRange:
                surrondingNumber = (2 * halfWidth + 1) ** 2
                avg = 0
                for ii in (i + rangeShift):
                    for jj in (j + rangeShift):
                        if (not (ii in rowRange)) or (not (jj in colRange)):
                            surrondingNumber -= 1
                        else:
                            avg += costFunction[ii, jj]
                newFunction[i, j] = avg * 1.0 / surrondingNumber
    return newFunction

def interp_map(img,npts=100.):
    fill=np.arange(img.shape[0]*npts)/npts
    x,y=np.meshgrid(fill,fill)
    finer=map_coordinates(img,[y,x])
    return finer
    
def search_center(img,x_cen,y_cen,size_window,m=0.2,M=0.8,size_cost=5,theta=[45,135],smooth=2,decimals=2,method=1,save='test'):
    (y_len,x_len) = img.shape
    x_range = np.arange(x_len)
    y_range = np.arange(y_len)
    
    if method==1:
        img_interp = interp2d(x_range,y_range,img,kind='cubic')
    else:
        img_interp = rbs(y_range,x_range,img)
    
    for decimal in range(decimals+1):
        precision = 10**(-decimal)
        
        x_centers = np.round(np.arange(x_cen-size_cost*precision,x_cen+size_cost*precision+precision*0.1,precision),decimals=decimal)
        y_centers = np.round(np.arange(y_cen-size_cost*precision,y_cen+size_cost*precision+precision*0.1,precision),decimals=decimal)
        cost_fun = np.zeros((x_centers.shape[0],y_centers.shape[0]))
        
        (xs,ys) = samplingRegion(size_window,theta,m,M)
        
        x_cen = 0.
        y_cen = 0.
        max_cost = 0.
        if method==1:
            for j,x0 in enumerate(x_centers):
                for i,y0 in enumerate(y_centers):
                    value=0.
                    for x1,y1 in zip(xs,ys):
                        x = x0+x1
                        y = y0+y1
                        value += img_interp(x,y)
                    cost_fun[i,j] = value
                    if value > max_cost:
                        max_cost = value
                        x_cen = x0
                        y_cen = y0
        else:
            for j,x0 in enumerate(x_centers):
                for i,y0 in enumerate(y_centers):
                    value=0.
                    for x1,y1 in zip(xs,ys):
                        x = x0+x1
                        y = y0+y1
                        value += img_interp(y,x)
                    cost_fun[i,j] = value
                    if value > max_cost:
                        max_cost = value
                        x_cen = x0
                        y_cen = y0
        plt.imshow(cost_fun,origin='lower')
        plt.savefig(save+'_costfn'+str(decimal)+'.svg')
        plt.clf()
        np.savez_compressed(save+'_costfn'+str(decimal),cost_fun=cost_fun)
    return x_cen,y_cen