# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 00:08:59 2018

@author: nnlab
"""
Data = np.loadtxt("RESPECT.txt",skiprows = 0)
L_list=Data[:,0]
T_list=Data[:,1]
R_list=Data[:,2]
d_min=1.0324999999993971



num=len(L_list)
n_s=np.arange(1,2,0.00001)
d_res=[]
T_res=[]
s=1
for n in n_s:
    for i in range (0,num):  
    A=16*s*n**2
    B=(n+s**2)*(n+1)**3
    C=2*(n**2- 1)*(n**2-s**2)
    D=(n-1)**3*(n-s**2)
    x=(1-R_list[i])**2/T_list[i]
    phi=4*np.pi*n*d_min*10**6/L_list[i] #--Z=Denominator of Tcal--
    phi_degree=math.degrees(phi)
    Z=B-C*x*np.cos(phi_degree*np.pi/180)+D*x 
    T_re = A*x/Z
    T_res.append(T_re)
    d_re =T_re-T_list[i]
    d_res.append(d_re) 
    print "n= %.4f,Tcal = %.7f,Tcal-T = %.10f"%(n,T_re,d_re)
    