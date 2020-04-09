# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:22:34 2020
Nathan Marshall

Coulomb explosion 3d with random orientation
"""
import numpy as np
from random import gauss
from scipy.integrate import solve_ivp
from numba import jit

file = 'S:/JRM_ARgroup/Nathan/Coltrims Sim/CO2' #file to export data to

num = 10000 #number of molecules to simulate

m1 = 12.011 #fragment masses
m2 = 15.999 
m3 = 15.999
amu_to_kg = 1.66053903e-27
m1, m2, m3 = m1 * amu_to_kg, m2 * amu_to_kg, m3 * amu_to_kg

q1 = 1 * 1.60217662e-19 #fragment charges 
q2 = 1 * 1.60217662e-19
q3 = 1 * 1.60217662e-19
k = 8.9875517923e9 #Coulomb force constant
V = 2000 #spectrometer voltage
L = 0.22 #spectrometer length in meters

vibmax = 1.16e-10 * 0.08 #maximum vibration amplitude for each fragment

r10 = np.array([0, 0, 0])  #fragment intial position vectors (x, y, z)
r20 = np.array([-1.16e-10, 0, 0]) 
r30 = np.array([1.16e-10, 0, 0]) 

v10 = np.array([0, 0, 0]) # fragment initial velocity vectors (vx, vy, vz)
v20 = np.array([0, 0, 0])
v30 = np.array([0, 0, 0])

vx10, vy10, vz10 = v10 #unpack fragment intial velocity vectors
vx20, vy20, vz20 = v20
vx30, vy30, vz30 = v30

t0 = 0           #start time
tmax = 5e-6     #stop time

tof1 = np.zeros(num) #arrays to store output data
x1 = np.zeros(num)
y1 = np.zeros(num)
tof2 = np.zeros(num)
x2 = np.zeros(num)
y2 = np.zeros(num)
tof3 = np.zeros(num)
x3 = np.zeros(num)
y3 = np.zeros(num)

@jit
def rand_vector():
    vec = np.zeros(3)
    for i in range(3):
        vec[i]= gauss(0, 1)
    vec = vec/np.linalg.norm(vec)
    return(vec)

@jit
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

@jit
def coul(r1, q1, m1, r2, q2):
    '''Accleration from force on charge q1 at r1 by charge q2 at r2.'''
    return k * q1 *q2 * (r1-r2) / np.linalg.norm(r1-r2)**3 / m1

@jit
def spec(q1, m1):
    '''Acceleration due to spectrometer.'''
    return q1 * (V/L) * np.array([0,0,1]) / m1

@jit
def diffeq(t, d):
    '''Differential equations to feed into solver.'''
    x1 = d[0] #define all observables
    y1 = d[1]
    z1 = d[2]
    x2 = d[3]
    y2 = d[4]
    z2 = d[5]
    x3 = d[6]
    y3 = d[7]
    z3 = d[8]
    r1 = np.array([x1, y1, z1])
    r2 = np.array([x2, y2, z2])
    r3 = np.array([x3, y3, z3])
    vx1 = d[9]
    vy1 = d[10]
    vz1 = d[11]
    vx2 = d[12]
    vy2 = d[13]
    vz2 = d[14]
    vx3 = d[15]
    vy3 = d[16]
    vz3 = d[17]
    
    #calculate accelerations of each fragment
    dvx1, dvy1, dvz1 = coul(r1, q1, m1, r2, q2) + coul(r1, q1, m1, r3, q3) + spec(q1, m1) 
    dvx2, dvy2, dvz2 = coul(r2, q2, m2, r1, q1) + coul(r2, q2, m2, r3, q3) + spec(q2, m2)
    dvx3, dvy3, dvz3 = coul(r3, q3, m3, r1, q1) + coul(r3, q3, m3, r2, q2) + spec(q3, m3)
    
    return(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, 
           dvx1, dvy1, dvz1, dvx2, dvy2, dvz2, dvx3, dvy3, dvz3)

def hit1(t, d): return L-d[2]
  
def hit2(t, d): return L-d[5]

def hit3(t, d): return L-d[8]

def simulate(r10, r20, r30):
    for i in range(num):
        axis = rand_vector()
        theta = 2 * np.pi * np.random.rand()
        
        r10_vib = r10 + rand_vector() * vibmax * np.random.rand()
        r20_vib = r20 + rand_vector() * vibmax * np.random.rand()
        r30_vib = r30 + rand_vector() * vibmax * np.random.rand()
        
        r10_vib = np.dot(rotation_matrix(axis, theta), r10_vib)
        r20_vib = np.dot(rotation_matrix(axis, theta), r20_vib)
        r30_vib = np.dot(rotation_matrix(axis, theta), r30_vib)
        x10, y10, z10 = r10_vib #unpack fragment initial position vectors
        x20, y20, z20 = r20_vib
        x30, y30, z30 = r30_vib
        
        #define initial values
        ivs = [x10, y10, z10, x20, y20, z20, x30, y30, z30, 
               vx10, vy10, vz10, vx20, vy20, vz20, vx30, vy30, vz30]
        
        #run differential equation solver with initial values
        sol = solve_ivp(diffeq, [t0, tmax], ivs, events=(hit1, hit2, hit3))
       
        #check for true detector hits and extract tof, x, and y
        if sol.t_events[0].size !=0 and sol.t_events[1].size !=0 and sol.t_events[2].size != 0:
            tof1[i] = sol.t_events[0][0]
            tof2[i] = sol.t_events[1][0]
            tof3[i] = sol.t_events[2][0]
            x1[i] = sol.y_events[0][0][0]
            y1[i] = sol.y_events[0][0][1]
            x2[i] = sol.y_events[1][0][3]
            y2[i] = sol.y_events[1][0][4]
            x3[i] = sol.y_events[2][0][6]
            y3[i] = sol.y_events[2][0][7]
          
def save_frags(file):
        delay = np.zeros(num)
        adc1 = np.zeros(num)
        adc2 = np.zeros(num)
        index = np.zeros(num)
        xyt_all = np.zeros((num, 13))
        xyt_all[:,0] = delay
        xyt_all[:,1] = x1*1000
        xyt_all[:,2] = y1*1000
        xyt_all[:,3] = tof1*1e9
        xyt_all[:,4] = x2*1000
        xyt_all[:,5] = y2*1000
        xyt_all[:,6] = tof2*1e9
        xyt_all[:,7] = x3*1000
        xyt_all[:,8] = y3*1000
        xyt_all[:,9] = tof3*1e9
        xyt_all[:,10] = adc1
        xyt_all[:,11] = adc2
        xyt_all[:,12] = index
        np.save(file, xyt_all)
        
simulate(r10, r20, r30)
save_frags(file)