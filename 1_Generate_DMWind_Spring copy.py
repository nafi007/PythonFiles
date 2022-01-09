#!/usr/bin/python3

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
import numpy.random
import scipy.stats as st
from scipy.stats import norm

 
#This comment is a test change 
################ ALEX DISTRIBUTION 2 (MOST RECENT) #############################
sigma = 110000 #110000

#Fall
#mu = 222036
#Spring
mu = 222036

def f1(v,theta,theta_prime,phi):
#     theta = np.arccos(cos_theta)    
    
    #NOTE we are using the cos_theta given to f1, not np.cos(theta) here
    sin_theta = np.sin(theta) 
    cos_theta = np.cos(theta)
    a = np.exp(-(v**2)/(2*(sigma**2)))
    b = np.exp(-1*v*mu*(cos_theta*np.cos(theta_prime) - sin_theta*np.sin(theta_prime)*np.cos(phi))/(sigma**2))
    c = np.sin(theta_prime) * cos_theta * sin_theta  #NOTE we are using the cos_theta given to f1, not np.cos(theta) here, as well as sin_theta that we computed
    result=((v**3)*a*b*c)
    return -1*result




# Draw random numbers from distribution

n = 200000

v1 = 0.0
v2 = 1100000
nv = 500

th1 = ((np.pi)/2.0) + (1e-15)
th2 = (np.pi) - (1e-15)
nth = 200

thp1 = 1e-15 # Needed for the function to not spit out incorrect value , 1e-15 is our "error tolerance"
thp2 = (np.pi) - 1e-15  # Needed for the function to not spit out incorrect value , 1e-15 is our "error tolerance"
nthp = 20

phi1 = 1e-15
phi2 = (2.0 * np.pi) - 1e-15
nphi = 20


# Pixel sizes in x and y
dv = (v2-v1)/np.real(nv)
dth = (th2-th1)/np.real(nth)
dthp = (thp2-thp1)/np.real(nthp)
dphi = (phi2-phi1)/np.real(nphi)

# Linearly spaced arrays of values corresponding to pixel centres
v = np.linspace(v1+(dv/2.0),v2-(dv/2.0),nv) 
th = np.linspace(th1+(dth/2.0),th2-(dth/2.0),nth)
thp = np.linspace(thp1+(dthp/2.0),thp2-(dthp/2.0),nthp)
phi = np.linspace(phi1+(dphi/2.0),phi2-(dphi/2.0),nphi)

# Make a grid of  v,th,thp,phi coordinate pairs
xyzt = np.asarray(np.meshgrid(v,th,thp,phi))

xyzt = xyzt.reshape(4,nv*nth*nthp*nphi) # Reshape the grid (4 here coresponds to 4 coordinates: v,th,thp,phi)

xyzt = np.transpose(xyzt).tolist() # Convert to a long list

# Make array of function values corresponding to the xyzt coordinates
V,TH,THP,PHI = np.meshgrid(v,th,thp,phi)
z = f1(V,TH,THP,PHI) # Array of function values

z = z.flatten() # Flatten array to create a long list of function values

z = z/sum(z) # Force normalisation, NOTE: if z is negative, then this removes the negative since sum(z) would be negative

# Make a list of integers linking xyzt coordinates to function values
i = list(range(z.size)) 

# Make the random choices with probabilties proportional to the function value
# The integer chosen by this can then be matched to the xyzt coordinates
j = np.random.choice(i,n,replace=True,p=z) 

# Now match the integers to the xy coordinates
vs = []
ths = []
thps = []
phis = []
for i in range(n):
    v_i,th_i,thp_i,phi_i = xyzt[j[i]]
    vs.append(v_i)
    ths.append(th_i)
    thps.append(thp_i)
    phis.append(phi_i)

# Random numbers for inter-pixel displacement
dvs = np.random.uniform(-dv/2.,dv/2.,n) 
dths = np.random.uniform(-dth/2.,dth/2.,n)
dthps = np.random.uniform(-dthp/2.,dthp/2.,n)
dphis = np.random.uniform(-dphi/2.,dphi/2.,n)

# Uniform-random displacement within a pixel
vs = vs+dvs
ths = ths+dths
thps = thps+dthps
phis = phis+dphis


MyList =  [vs,ths,thps,phis]
MyList = np.asarray(MyList)
MyList = MyList.T

np.save("Feb_200k_InitPar_Spring",MyList)