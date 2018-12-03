#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:46:33 2018
@author: chunglee_people
"""
import numpy as np 
#--------------initial condition-----------
n=1.369
s=1
l=590         ##nm
R=0.055339
T=0.85808
#--------------Tcal's parameters-----------
A=16*s*n**2
B=(n+s**2)*(n+1)**3
C=2*(n**2- 1)*(n**2-s**2)
D=(n-1)**3*(n-s**2)
#--------------find x----------------------
x=(1-R)**2/T
#-------------Iteration angle------------
angles=np.arange(0,361,1)
angles_p=[]
T_cals=[]
deltas=[]
thicks=[]
for angle in angles:
    Z=B-C*x*np.cos(angle*np.pi/180)+D*x  #--Z=Denominator of Tcal--
    angles_p.append(Z)   
    
    T_cal = A*x/Z
    T_cals.append(T_cal)
    
    delta = T - T_cal
    deltas.append(delta)      
    
    thick=angle*l/(4*np.pi*n)
    thicks.append(thick)
    print "Tcal = %.4f,T-Tcal = %.4f, d = %.4f"%(T_cal,delta,thick)
#-------------write into text---------------
string="%8s\t%8s\t%8s\t%8s\n"%("NO.","Tcal","T-Tcal","d")
for i in range(360):
   string+="%8d\t%f\t%f\t%f\n"%(i+1,T_cals[i],deltas[i],thicks[i])
f = open("TRI590.txt","w")
f.write(string)
f.close()
#-------------Tcal's formula---------------






































