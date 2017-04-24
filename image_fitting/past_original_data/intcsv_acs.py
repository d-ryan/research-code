import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv

location='F:/BetaPic/acs/raw/'
prefix='hst_9861_'
numlist=['01','04','05','06','07','08','09','10']
suffix='_acs_wfc_f'
wvlist=['435','606','814']
suffix2='w_drz.fits'

pairs=[(0,1),(0,2),(1,2),(0,),(2,),(1,),(0,2),(1,2)]

method = ['lin','parab','cubic','ave']

side=['LEFT','RIGHT','TOP','BOT']

lx=[]
px=[]
cx=[]
ly=[]
py=[]
cy=[]
avex=[]
avey=[]
ns=[]
ws=[]
ms=[]


if __name__=='__main__':
    for i,n in enumerate(numlist): #Image number
        for j in pairs[i]: #Wavelength

            with np.load(location+n+'/intdata.npz') as d:
                ints=d['ints']
            lx.append(ints[0,0])
            ly.append(ints[0,1])
            px.append(ints[1,0])
            py.append(ints[1,1])
            cx.append(ints[2,0])
            cy.append(ints[2,1])
            avex.append((ints[0,0]+ints[1,0]+ints[2,0])/3.)
            avey.append((ints[0,1]+ints[1,1]+ints[2,1])/3.)
            ns.append(n)
            ws.append(wvlist[j])

    with open('acs_ints.csv','wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Num.','Wavelength','Method','x','y','x-avex','y-avey'])
        for i in range(len(lx)):
            writer.writerow([ns[i],ws[i],method[0],lx[i],ly[i],lx[i]-avex[i],ly[i]-avey[i]])
            writer.writerow([ns[i],ws[i],method[1],px[i],py[i],px[i]-avex[i],py[i]-avey[i]])
            writer.writerow([ns[i],ws[i],method[2],cx[i],cy[i],cx[i]-avex[i],cy[i]-avey[i]])
            writer.writerow([ns[i],ws[i],method[3],avex[i],avey[i],0,0])