import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from spikefit_gen_dev import get_data
import pdb
import csv


with np.load('sir_B_cens.npz') as d:
    results = d['results']
    fnames = d['fnames']

with open('sir_B_cens.csv','wb') as csvfile:
    writer = csv.writer(csvfile)
    row = ['File','x','y']
    writer.writerow(row)
    for i in range(16):
        name = fnames[i]
        result = results[i]
        row = [name,result[0],result[1]]
        writer.writerow(row)