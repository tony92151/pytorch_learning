#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import math

try:
    Data = np.loadtxt("data.csv", dtype=np.str, delimiter=",")
    L_list=Data[1:,0].astype(np.float)
    T_list=Data[1:,1].astype(np.float)
    R_list=Data[1:,2].astype(np.float)
    #L = 2*(np.ones(len(L_list)))
    #print(L_list-L)
except:
    pritn("Load file faild!")
    sys.exit(0)

#--------------initial condition-----------
s = 1
l = 590         ##nm
d = 1.032699999

#--------------initial condition-----------
s = 1
l = 590         ##nm
d = 1.032699999

#--------------Tcal's parameters-----------
n = np.linspace(0.4, 1.8, len(L_list))

A = 16*s*(n**2)
B = (n+s**2)*(n+1)**3
C = 2*(n**2- 1)*(n**2-s**2)
D = (n-1)**3*(n-s**2)
x=(1-R_list)**2/T_list

DT = ((A*x)/(B-C*x*np.cos((4*math.pi*n*d)/(l)*(180/math.pi)*math.pow(10,-6))+D))-(T_list)
#print("length of n = %f" %len(n))
#print("length of DT = %f" %len(DT))
plt.scatter(n,DT, marker='o', lw=0.1,label='Tcal-T')
plt.xlabel('n')
plt.ylabel('Tcal-T')
plt.show()
#time.sleep(10)