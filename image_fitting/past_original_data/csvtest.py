import numpy as np
import csv

with open('testingcsv.csv','wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([3,7,'=A1+A2'])