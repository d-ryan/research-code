import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.interactive(True)
from scipy.ndimage import map_coordinates

class register(object):
    def __init__(self,img1,img2,cen1,cen2):
        self.img1 = img1
        self.orig2 = img2
        self.img2 = img2
        self.cen1 = cen1
        self.cen2 = cen2
        self.testcen = cen2
        self.default_inc = 0.1
        self.fig = plt.figure()
        self.fig.set_size_inches(12,12,forward=True)

    def all_stack(self):
        x1 = self.cen1[0]
        x2 = self.testcen[0]
        y1 = self.cen1[1]
        y2 = self.testcen[1]
        dx = x2-x1
        dy = y2-y1
        coords = np.ones((2,)+self.img1.shape)
        xs = np.arange(self.img1.shape[1])
        ys = np.arange(self.img1.shape[0])
        y_ones = np.ones(self.img1.shape[0])
        x_ones = np.ones(self.img1.shape[1])
        coords[0,:,:] = np.outer(ys,x_ones)+dy
        coords[1,:,:] = np.outer(y_ones,xs)+dx
        
        self.shifted = map_coordinates(self.img2,coords,mode='nearest')
        self.stacked = self.img1+self.shifted
        self.diffed = self.img1-self.shifted
    '''    
    def shift(self,img1,img2,cen1,cen2):
        shifted, stacked, diffed = all_stack(img1,img2,cen1,cen2)
        return shifted
        
    def stack(img1,img2,cen1,cen2):
        shifted, stacked, diffed = all_stack(img1,img2,cen1,cen2)
        return stacked
        
    def diff(img1,img2,cen1,cen2):
        shifted, stacked, diffed = all_stack(img1,img2,cen1,cen2)
        return diffed
    '''
    def scale_max(self):
        max1 = self.img1.max()
        max2 = self.img2.max()
        self.img2 = self.img2/max2*max1
    def scale_mean(self):
        mean1 = np.mean(self.img1)
        mean2 = np.mean(self.img2)
        self.img2 = self.img2/mean2*mean1
    
    def reset_cen2(self):
        self.testcen = self.cen2
        self.all_stack()
        
    def L(self,inc=None):
        if inc is None: inc = self.default_inc
        tx = self.testcen[0]
        ty = self.testcen[1]
        tx += inc
        self.testcen = (tx,ty)
        self.update_plot()
    def U(self,inc=None):
        if inc is None: inc = self.default_inc
        tx = self.testcen[0]
        ty = self.testcen[1]
        ty -= inc
        self.testcen = (tx,ty)
        self.update_plot()
    def D(self,inc=None):
        if inc is None: inc = self.default_inc
        tx = self.testcen[0]
        ty = self.testcen[1]
        ty += inc
        self.testcen = (tx,ty)
        self.update_plot()
    def R(self,inc=None):
        if inc is None: inc = self.default_inc
        tx = self.testcen[0]
        ty = self.testcen[1]
        tx -= inc
        self.testcen = (tx,ty)
        self.update_plot()
    def set_plot_img(self,img):
        self.plot_img = img
    def update_plot(self):
        self.all_stack()
        self.set_plot_img(self.diffed)
        plt.figure(self.fig.number)
        plt.clf()
        plotter = self.plot_img-self.plot_img.min()+0.0001*self.plot_img.max()
        low = np.percentile(plotter,20)
        high = np.percentile(plotter,80)
        plotter[plotter<low]=low
        plotter[plotter>high]=high
        plt.imshow(np.log(plotter),origin='lower',cmap=plt.cm.hot)