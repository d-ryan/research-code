import numpy as np
import matplotlib.pyplot as plt
import os

def comp_results(prefix):

    fs = os.listdir(prefix)
    num = len(fs)

    vars = np.zeros((18,num))
    lin = np.zeros((2,num))
    par = np.zeros((2,num))
    cub = np.zeros((2,num))

    for i,filename in enumerate(fs):
        with np.load(prefix+filename) as d:
            if i==0:
                n = d['used']
                s = d['s']
            ints = d['ints']
            cen = d['cen']
            cov = d['cov']
            param = d['param']
        for j in range(18):
            if j<4:
                w=j%2
                b=j//2
            elif j<10:
                w=(j-4)%3
                b=(j-4)//3+2
            else:
                w=(j-10)%4
                b=(j-10)//4+4
            vars[j,i]=cov[w,w,b]
        lin[:,i] = ints[0,:]-cen
        par[:,i] = ints[1,:]-cen
        cub[:,i] = ints[2,:]-cen
    savename = 'F:/error_prop/data_products/parab/'+str(n)+'_'+str(s)
    np.savez_compressed(savename,vars=vars,lin=lin,par=par,cub=cub)
    return savename
    
def consolidate(comp_dir):
    comp_dir_listed = os.listdir(comp_dir)
    for z in comp_dir_listed[:]:
        if z[-3:]!='npz': comp_dir_listed.remove(z)
    ns = np.linspace(100,1500,15).astype('int')
    ss = np.linspace(0.02,0.3,29)
    xs = np.zeros((15,29))
    ys = np.zeros((15,29))
    for i,comp_name in enumerate(comp_dir_listed):
        #if i>1: break
        splitstr = comp_name.split('_')
        n = int(splitstr[0])
        s = float(splitstr[1][:-4])
        #print n
        nw = np.where(ns==n)[0][0]
        #print n
        
        #print s
        #print ss
        #print np.where(np.round(ss,6)==s)
        #print list(ss)
        #print list(ss).index(s)
        sw = np.where(np.round(ss,6)==s)[0][0]
        with np.load(comp_dir+'/'+comp_name) as d:
            vars = d['vars']
            dat = d['cub']
        x = dat[0,:]
        y = dat[1,:]
        x_m = np.mean(x)
        y_m = np.mean(y)
        x_std = np.std(x)
        y_std = np.std(y)
        xs[nw,sw]=x_m
        ys[nw,sw]=y_m
    #print np.where(xs==0)
    #print np.where(ys==0)
    #plt.imshow(xs)
    #plt.show()
    #plt.imshow(ys)
    #plt.show()
    plot_s = [0.02,0.05,0.08,0.1,0.15,0.2,0.25,0.3]
    j = 0
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    #plt.xlabel('Pixels used in linear fit, per spike')
    plt.ylabel('Centering error in x [pix]')
    plt.title('Systematic error in cubic fit to distorted spikes')
    #plt.yscale('log')
    #plt.xscale('log')
    ax2 = fig.add_subplot(212)
    plt.xlabel('Pixels used in cubic fit, per spike')
    plt.ylabel('Centering error in y [pix]')
    #plt.yscale('log')
    #plt.xscale('log')
    for i,s in enumerate(ss):
        if np.round(s,6) in plot_s:
            col = plt.cm.viridis.colors[j]
            j+=35
            ax1.plot(ns,xs[:,i],color=col,label=r'$\sigma$ = '+str(np.round(s,6)))
            #ax1.plot(ns,s/np.sqrt(ns),color=col,linestyle=':')
            ax2.plot(ns,ys[:,i],color=col,label=r'$\sigma$ = '+str(np.round(s,6)))
            #ax2.plot(ns,s/np.sqrt(ns),color=col,linestyle=':')
    #ax1.legend()
    #ax2.legend()
    plt.show()
    #np.savez_compressed('F:/error_prop/data_products/error_matrix',ns=ns,ss=ss,xs=xs,ys=ys)
    
if __name__=='__main__':
    consolidate('F:/error_prop/data_products/cubic/')
        
        
    
        
    
def plot_results(comp_name):

    with np.load(comp_name) as d:
        vars = d['vars']
        lin = d['lin']
        par = d['par']
        cub = d['cub']
        
    fig = plt.figure()
    ax1 = fig.add_subplot(321)
    n,bins,patches = ax1.hist(lin[0,:],bins=30,normed=1)
    mu = np.mean(lin[0,:])
    sig = np.std(lin[0,:])
    l, = ax1.plot(bins,1./(sig*np.sqrt(2.*np.pi))*np.exp(-(bins-mu)**2./(2*sig**2.)),color='r')
    plt.xlabel('Linear fit, x error')
    ax2 = fig.add_subplot(322)
    n,bins,patches = ax2.hist(lin[1,:],bins=30,normed=1)
    mu = np.mean(lin[1,:])
    sig = np.std(lin[1,:])
    l, = ax2.plot(bins,1./(sig*np.sqrt(2.*np.pi))*np.exp(-(bins-mu)**2./(2*sig**2.)),color='r')
    plt.xlabel('Linear fit, y error')
    ax3 = fig.add_subplot(323)
    n,bins,patches = ax3.hist(par[0,:],bins=30,normed=1)
    mu = np.mean(par[0,:])
    sig = np.std(par[0,:])
    l, = ax3.plot(bins,1./(sig*np.sqrt(2.*np.pi))*np.exp(-(bins-mu)**2./(2*sig**2.)),color='r')
    plt.xlabel('Parab fit, x error')
    ax4 = fig.add_subplot(324)
    n,bins,patches = ax4.hist(par[1,:],bins=30,normed=1)
    mu = np.mean(par[1,:])
    sig = np.std(par[1,:])
    l, = ax4.plot(bins,1./(sig*np.sqrt(2.*np.pi))*np.exp(-(bins-mu)**2./(2*sig**2.)),color='r')
    plt.xlabel('Parab fit, y error')
    ax5 = fig.add_subplot(325)
    n,bins,patches = ax5.hist(cub[0,:],bins=30,normed=1)
    mu = np.mean(cub[0,:])
    sig = np.std(cub[0,:])
    l, = ax5.plot(bins,1./(sig*np.sqrt(2.*np.pi))*np.exp(-(bins-mu)**2./(2*sig**2.)),color='r')
    plt.xlabel('Cubic fit, x error')
    ax6 = fig.add_subplot(326)
    n,bins,patches = ax6.hist(cub[1,:],bins=30,normed=1)
    mu = np.mean(cub[1,:])
    sig = np.std(cub[1,:])
    l, = ax6.plot(bins,1./(sig*np.sqrt(2.*np.pi))*np.exp(-(bins-mu)**2./(2*sig**2.)),color='r')
    plt.xlabel('Cubic fit, y error')
    plt.savefig(comp_name+'.pdf')
    plt.savefig(comp_name+'.svg')
    plt.savefig(comp_name+'.png')
    plt.show()