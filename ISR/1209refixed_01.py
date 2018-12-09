
import numpy as np
#l-> R and  T
def read_R(l):
    Data = np.loadtxt("RESPECT_test.txt",skiprows = 0)
    L_list=Data[:,0]      
    T_list=Data[:,1]      
    R_list=Data[:,2]      
    num=len(L_list)
    row = 0
    for j in range(num):
        if L_list[j] == l:
            row = j
    return R_list[row]

def read_T(l):
    Data = np.loadtxt("RESPECT_test.txt",skiprows = 0)
    L_list=Data[:,0]      
    T_list=Data[:,1]      
    R_list=Data[:,2]      
    num=len(L_list)
    row = 0
    for j in range(num):
        if L_list[j] == l:
            row = j
    return T_list[row]


def find_nearest(d,T_cal,delta):
    idx = (np.abs(delta)).argmin()
    return (d[idx], T_cal[idx], delta[idx])

"""
def find_nearest_n(T_cal,delta):
    idx = (np.abs(delta)).argmin()
    return (T_cal[idx], delta[idx])
"""


def getd(n,s,l,R,T):
    T_cals=[]
    deltas=[]
    A=16*s*n**2
    B=(n+s**2)*(n+1)**3
    C=2*(n**2- 1)*(n**2-s**2)
    D=(n-1)**3*(n-s**2)
    x=(1-R)**2/T
    d_s=np.arange(0.9,1.1,0.00001)
    #-----------------------------
    for d in d_s:
        phi=4*np.pi*n*d*10**6/l #T_cal's_angle 
        Z=B-C*x*np.cos(phi*np.pi/180)+D*x 
        T_cal = A*x/Z         #T_cal's_formula
        T_cals.append(T_cal)  
        delta = T_cal-T       
        deltas.append(delta) 
        #print "d= %.4f,Tcal = %.7f,Tcal-T = %.10f"%(d,T_cal,delta)
    min_val = find_nearest(d_s,T_cals,deltas)
    #print "d_min= %.4f,Tcal = %.7f,Tcal-T = %.10f"%(min_val[0],min_val[1],min_val[2])
    #print min_val[0]
    return (min_val[0],min_val[1],min_val[2])

def getT(n,s,l,R,T,d):
    #T_cals=[]
    #deltas=[]
    A=16*s*n**2
    B=(n+s**2)*(n+1)**3
    C=2*(n**2- 1)*(n**2-s**2)
    D=(n-1)**3*(n-s**2)
    x=(1-R)**2/T
    #-----------------------------
    phi=4*np.pi*n*d*10**6/l #T_cal's_angle 
    Z=B-C*x*np.cos(phi*np.pi/180)+D*x 
    T_cal = A*x/Z         #T_cal's_formula
    #T_cals.append(T_cal)  
    #delta = T_cal-T       
    #deltas.append(delta) 
    
    return T_cal



def find_new_n(l,l_next,R,T,n,d_c):
    n_upper_array = np.arange(n,n+0.001,0.0001)
    n_lower_array = np.arange(n-0.001,n,0.0001)
    lu = len(n_upper_array)
    ll = len(n_lower_array)
    s = 1
    delta_array = []
    T_cal_array = []
    if l_next > l:
        #n_upper_array
        for i in range(lu):
            temp = getT(n_upper_array[i],s,l_next,read_R(l_next),read_T(l_next),d_c)  #s=1
            delta_array.append(np.abs(temp - T))
            #d_array.append(temp[0])
            T_cal_array.append(temp)
        min_delta = np.amin(delta_array)
        p = np.where(delta_array == min_delta)
        p0 = p[0]
        #print delta_array
        #print min_delta
        #print p[0]
        new_n = n_upper_array[p0]
    elif l_next < l:
        #n_lower_array
        for i in range(ll):
            temp = getT(n_lower_array[i],s,l_next,read_R(l_next),read_T(l_next),d_c)  #s=1
            delta_array.append(np.abs(temp - T))
            #d_array.append(temp[0])
            T_cal_array.append(temp)
        min_delta = np.amin(delta_array)
        p = np.where(delta_array == min_delta)
        p0 = p[0]
        #print delta_array
        #print min_delta
        new_n = n_lower_array[p0]
    else:
        new_n = [n]
    return new_n

l_array = np.linspace(585,595,11)
#l_array = [591]

temp2 = getd(1.369,1,590,read_R(590),read_T(590))

New_n = 1.369

for k in range(11):
    print l_array[k]
    if l_array[k] >= 591:
        New_n = find_new_n(590,l_array[k],read_R(l_array[k]),read_T(l_array[k]),New_n,temp2[0])
        print New_n 
    else:
        Old_n = find_new_n(590,l_array[k],read_R(l_array[k]),read_T(l_array[k]),New_n,temp2[0])
        print Old_n 
