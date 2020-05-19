# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:59:31 2020
Nathan Marshall

4-body Coulomb explosion 3d with random orientation
"""
import numpy as np
from random import gauss
from scipy.integrate import solve_ivp
from numba import jit

file = 'C:/Users/Nathan/Desktop/CHD_sim2' #file to export data to

num = 10000 #number of molecules to simulate

m1 = 14.027 #fragment masses in g/mol
m2 = 14.027 
m3 = 26.038
m4 = 26.038
gmol_to_kg = 1.66053903e-27 #convert mass from g/mole to kg
m1, m2, m3, m4 = m1 * gmol_to_kg, m2 * gmol_to_kg, m3 * gmol_to_kg, m4 * gmol_to_kg

q1 = 1 * 1.60217662e-19 #fragment charges 
q2 = 1 * 1.60217662e-19
q3 = 1 * 1.60217662e-19
q4 = 1 * 1.60217662e-19
k = 8.9875517923e9 #Coulomb force constant
V = 2000 #spectrometer voltage
L = 0.22 #spectrometer length in meters

vibmax1 = 1.54e-10 *0.08 #maximum vibration amplitude for each fragment
vibmax2 = 1.54e-10 *0.08
vibmax3 = 1.54e-10 *0.08
vibmax4 = 1.54e-10 *0.08

#fragment intial position vectors (x, y, z)
r10 = np.array([1.6387845757, 0.3062579488, -0.3362671360])*1e-10
r20 = np.array([-1.0963371422, 1.2363558682, 0.3378773075])*1e-10
r30 = np.array([0.3511198941, -1.7705884636, 0.5082857954])*1e-10 
r40 = np.array([-1.3445103983, -1.1865339385, -0.5296801941])*1e-10

v10 = np.array([0, 0, 0]) # fragment initial velocity vectors (vx, vy, vz)
v20 = np.array([0, 0, 0])
v30 = np.array([0, 0, 0])
v40 = np.array([0, 0, 0])

vx10, vy10, vz10 = v10 #unpack fragment initial velocity vectors
vx20, vy20, vz20 = v20
vx30, vy30, vz30 = v30
vx40, vy40, vz40 = v40

t0 = 0           #integration start time
tmax = 5e-6      #integration stop time

tof1 = np.zeros(num) #create arrays to store output data
x1 = np.zeros(num)
y1 = np.zeros(num)
tof2 = np.zeros(num)
x2 = np.zeros(num)
y2 = np.zeros(num)
tof3 = np.zeros(num)
x3 = np.zeros(num)
y3 = np.zeros(num)
tof4 = np.zeros(num)
x4 = np.zeros(num)
y4 = np.zeros(num)

@jit
def rand_vector():
    '''Generates a spherically uniform random unit vector.'''
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
    '''Accleration from Coulomb force on charge q1 at r1 by charge q2 at r2.'''
    return k * q1 *q2 * (r1-r2) / np.linalg.norm(r1-r2)**3 / m1

@jit
def spec(q1, m1):
    '''Acceleration due to spectrometer on charge q1.'''
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
    x4 = d[9]
    y4 = d[10]
    z4 = d[11]
    r1 = np.array([x1, y1, z1]) #define current position vector
    r2 = np.array([x2, y2, z2])
    r3 = np.array([x3, y3, z3])
    r4 = np.array([x4, y4, z4])
    vx1 = d[12]
    vy1 = d[13]
    vz1 = d[14]
    vx2 = d[15]
    vy2 = d[16]
    vz2 = d[17]
    vx3 = d[18]
    vy3 = d[19]
    vz3 = d[20]
    vx4 = d[21]
    vy4 = d[22]
    vz4 = d[23]
    
    #calculate accelerations of each fragment
    dvx1, dvy1, dvz1 = coul(r1, q1, m1, r2, q2) + coul(r1, q1, m1, r3, q3) + coul(r1, q1, m1, r4, q4) + spec(q1, m1) 
    dvx2, dvy2, dvz2 = coul(r2, q2, m2, r1, q1) + coul(r2, q2, m2, r3, q3) + coul(r2, q2, m2, r4, q4) + spec(q2, m2)
    dvx3, dvy3, dvz3 = coul(r3, q3, m3, r1, q1) + coul(r3, q3, m3, r2, q2) + coul(r3, q3, m3, r4, q4) + spec(q3, m3)
    dvx4, dvy4, dvz4 = coul(r4, q4, m4, r1, q1) + coul(r4, q4, m4, r2, q2) + coul(r4, q4, m4, r3, q3) + spec(q4, m4)
    
    return(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, vx4, vy4, vz4,
           dvx1, dvy1, dvz1, dvx2, dvy2, dvz2, dvx3, dvy3, dvz3, dvx4, dvy4, dvz4)

def hit1(t, d): return L-d[2] #functions for detecting spectrometer hits
  
def hit2(t, d): return L-d[5]

def hit3(t, d): return L-d[8]

def hit4(t, d): return L-d[11]

def simulate(r10, r20, r30, r40):
    '''Runs the simulation for a given number of molecules.'''
    for i in range(num):
        
        #add a random vibration to initial position vectors 
        r10_vib = r10 + rand_vector() * vibmax1 * np.random.rand()
        r20_vib = r20 + rand_vector() * vibmax2 * np.random.rand()
        r30_vib = r30 + rand_vector() * vibmax3 * np.random.rand()
        r40_vib = r40 + rand_vector() * vibmax4 * np.random.rand()
        
        axis = rand_vector() #choose random axis
        theta = 2 * np.pi * np.random.rand() #choose random angle
        
        #rotate the position vectors around random axis by random angle
        r10_vib = np.dot(rotation_matrix(axis, theta), r10_vib)
        r20_vib = np.dot(rotation_matrix(axis, theta), r20_vib)
        r30_vib = np.dot(rotation_matrix(axis, theta), r30_vib)
        r40_vib = np.dot(rotation_matrix(axis, theta), r40_vib)
        
        x10, y10, z10 = r10_vib #unpack fragment initial position vectors
        x20, y20, z20 = r20_vib
        x30, y30, z30 = r30_vib
        x40, y40, z40 = r40_vib
        
        #define initial conditions list for the diffeq solver
        ivs = [x10, y10, z10, x20, y20, z20, x30, y30, z30, x40, y40, z40,
               vx10, vy10, vz10, vx20, vy20, vz20, vx30, vy30, vz30, vx40, vy40, vz40]
        
        #run differential equation solver with initial values
        sol = solve_ivp(diffeq, [t0, tmax], ivs, events=(hit1, hit2, hit3, hit4))
       
        #check for true detector hits and extract tof, x, and y
        #if sol.t_events[0].size !=0 and sol.t_events[1].size !=0 and sol.t_events[2].size != 0 and sol.t_events[3] != 0:
        tof1[i] = sol.t_events[0][0]
        tof2[i] = sol.t_events[1][0]
        tof3[i] = sol.t_events[2][0]
        tof4[i] = sol.t_events[3][0]
        x1[i] = sol.y_events[0][0][0]
        y1[i] = sol.y_events[0][0][1]
        x2[i] = sol.y_events[1][0][3]
        y2[i] = sol.y_events[1][0][4]
        x3[i] = sol.y_events[2][0][6]
        y3[i] = sol.y_events[2][0][7]
        x4[i] = sol.y_events[3][0][9]
        y4[i] = sol.y_events[3][0][10]
          
def save_data(file):
    '''Save X, Y, and TOF data to a binary file.'''
    delay = np.zeros(num) #zero array placeholders
    adc1 = np.zeros(num)
    adc2 = np.zeros(num)
    index = np.zeros(num)
    xyt_all = np.zeros((num, 16)) #array to save as binary
    xyt_all[:,0] = delay
    xyt_all[:,1] = x1*1000  #convert from m to mm
    xyt_all[:,2] = y1*1000
    xyt_all[:,3] = tof1*1e9 #convert from s to ns
    xyt_all[:,4] = x2*1000
    xyt_all[:,5] = y2*1000
    xyt_all[:,6] = tof2*1e9
    xyt_all[:,7] = x3*1000
    xyt_all[:,8] = y3*1000
    xyt_all[:,9] = tof3*1e9
    xyt_all[:,10] = x4*1000
    xyt_all[:,11] = y4*1000
    xyt_all[:,12] = tof4*1e9
    xyt_all[:,13] = adc1
    xyt_all[:,14] = adc2
    xyt_all[:,15] = index
    np.save(file, xyt_all) #save array as a binary file
        
simulate(r10, r20, r30, r40) #run simulator with given initial positions
save_data(file)        #save the data