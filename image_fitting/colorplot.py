import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import viridis

#colormap syntax:
#colormap_name(x,alpha=None)
#x : if float, should be w/in [0.,1.]
#x : if int, should be w/in [0,colormap_name.N]
# returns tuple of RGBA values

def colorplot(xvar,yvar,colorvar,axes,cmap=viridis,**kwargs):
    
    
    colornorm = Normalize(colorvar)
    sc = axes.scatter(xvar,yvar,)