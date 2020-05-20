# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:24:33 2020
Nathan Marshall

Three-Body Coulomb explosion with sequential fragmentation.
"""
import numpy as np
from random import gauss
from scipy.integrate import solve_ivp
from numba import jit

file = 'S:/JRM_ARgroup/Nathan/Coltrims Sim/CO2_sequential' #file to export data to

num = 1000 #number of molecules to simulate

m11 = 15.999 #fragment masses before sequential fragmentation
m21 = 28.010

m12 = 15.999 #fragment masses after sequential fragmentation
m22 = 12.011 
m32 = 15.999

gmol_to_kg = 1.66053903e-27 #convert mass from g/mole to kg
m11, m21 = m11 * gmol_to_kg, m21 * gmol_to_kg
m12, m22, m32 = m12 * gmol_to_kg, m22 * gmol_to_kg, m32 * gmol_to_kg

Q = 1.60217662e-19
q11 = 1 * Q #fragment charges before sequential fragmentation
q21 = 2 * Q

q12 = 1 * Q #fragment charges after sequential fragmentation
q22 = 1 * Q
q32 = 1 * Q

k = 8.9875517923e9 #Coulomb force constant
V = 2000 #spectrometer voltage
L = 0.22 #spectrometer length in meters

vibmax = 1.16e-10 * 0 #maximum vibration amplitude for each fragment

r11 = np.array([-1.16e-10, 0, 0])  #2 body intial position vectors (x, y, z)
r21 = np.array([0.683e-10, 0, 0]) 
  
r22 = np.array([0, 0, 0]) #fragment intial position vectors (x, y, z)
r32 = np.array([1.16e-10, 0, 0]) 

v11 = np.array([0, 0, 0]) # fragment initial velocity vectors (vx, vy, vz)
v21 = np.array([0, 0, 0])

vx11, vy11, vz11 = v11 #unpack fragment intial velocity vectors
vx21, vy21, vz21 = v21

tof1 = np.zeros(num) #create arrays to store output data
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
    '''Acceleration from Coulomb force on charge q1 at r1 by charge q2 at r2.'''
    return k * q1 *q2 * (r1-r2) / np.linalg.norm(r1-r2)**3 / m1

@jit
def spec(q1, m1):
    '''Acceleration due to spectrometer on charge q1.'''
    return q1 * (V/L) * np.array([0,0,1]) / m1

@jit
def diffeq1(t, d):
    '''Differential equations to feed into solver.'''
    x1 = d[0] #define all observables
    y1 = d[1]
    z1 = d[2]
    x2 = d[3]
    y2 = d[4]
    z2 = d[5]
    r1 = np.array([x1, y1, z1]) #define current position vector
    r2 = np.array([x2, y2, z2])
    vx1 = d[6]
    vy1 = d[7]
    vz1 = d[8]
    vx2 = d[9]
    vy2 = d[10]
    vz2 = d[11]
    
    #calculate accelerations of each fragment
    dvx1, dvy1, dvz1 = coul(r1, q11, m11, r2, q21) + spec(q11, m11) 
    dvx2, dvy2, dvz2 = coul(r2, q21, m21, r1, q11) + spec(q21, m21)
    
    return(vx1, vy1, vz1, vx2, vy2, vz2, dvx1, dvy1, dvz1, dvx2, dvy2, dvz2)

@jit
def diffeq2(t, d):
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
    r1 = np.array([x1, y1, z1]) #define current position vector
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
    dvx1, dvy1, dvz1 = coul(r1, q12, m12, r2, q22) + coul(r1, q12, m12, r3, q32) + spec(q12, m12) 
    dvx2, dvy2, dvz2 = coul(r2, q22, m22, r1, q12) + coul(r2, q22, m22, r3, q32) + spec(q22, m22)
    dvx3, dvy3, dvz3 = coul(r3, q32, m32, r1, q12) + coul(r3, q32, m32, r2, q22) + spec(q32, m32)
    
    return(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, 
           dvx1, dvy1, dvz1, dvx2, dvy2, dvz2, dvx3, dvy3, dvz3)

def hit1(t, d): return L-d[2] #functions for detecting spectrometer hits
  
def hit2(t, d): return L-d[5]

def hit3(t, d): return L-d[8]

def simulate(r11, r21, r22, r32):
    '''Runs the simulation for a given number of molecules.'''
    for i in range(num):
        
        #add a random vibration to initial position vectors 
        r11_vib = r11 + rand_vector() * vibmax * np.random.rand()
        r21_vib = r21 + rand_vector() * vibmax * np.random.rand()
        
        axis = rand_vector() #choose random axis
        theta = 2 * np.pi * np.random.rand() #choose random angle
        
        #rotate the position vectors around random axis by random angle
        r11_vib = np.dot(rotation_matrix(axis, theta), r11_vib)
        r21_vib = np.dot(rotation_matrix(axis, theta), r21_vib)
        
        x11, y11, z11 = r11_vib #unpack fragment initial position vectors
        x21, y21, z21 = r21_vib
        
        #define initial conditions list for the diffeq solver
        ivs = [x11, y11, z11, x21, y21, z21, 0, 0, 0, 0, 0, 0]
        
        t0 = 0 #integration start time
        tmax = np.random.normal(loc=20e-9, scale=15e-9) #integration stop time
        
        #run differential equation solver with initial values
        sol = solve_ivp(diffeq1, [t0, tmax], ivs)
        
        r11 = np.array([sol.y[0][-1], sol.y[1][-1], sol.y[2][-1]])
        r21 = np.array([sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]])
        v11 = np.array([sol.y[6][-1], sol.y[7][-1], sol.y[8][-1]])
        v21 = np.array([sol.y[9][-1], sol.y[10][-1], sol.y[11][-1]])
        
        axis = rand_vector() #choose random axis
        theta = 2 * np.pi * np.random.rand() #choose random angle
        
        r22 = np.dot(rotation_matrix(axis, theta), r22)
        r32 = np.dot(rotation_matrix(axis, theta), r32)
        
        r12 = r11
        r22 = r22 + r21
        r32 = r32 + r21
        
        x12, y12, z12 = r12 #unpack fragment initial position vectors
        x22, y22, z22 = r22
        x32, y32, z32 = r32
        
        vx11, vy11, vz11 = v11
        vx21, vy21, vz21 = v21
        
        #define initial conditions list for the diffeq solver
        ivs = [x12, y12, z12, x22, y22, z22, x32, y32, z32, 
               vx11, vy11, vz11, vx21, vy21, vz21, vx21, vy21, vz21]
        
        t0 = tmax #integration start time
        tmax = 3.5e-6 #integration stop time
        
        #run differential equation solver with initial values
        sol = solve_ivp(diffeq2, [t0, tmax], ivs, events=(hit1, hit2, hit3))
        
        #check for true detector hits and extract tof, x, and y
        tof1[i] = sol.t_events[0][0]
        tof2[i] = sol.t_events[1][0]
        tof3[i] = sol.t_events[2][0]
        x1[i] = sol.y_events[0][0][0]
        y1[i] = sol.y_events[0][0][1]
        x2[i] = sol.y_events[1][0][3]
        y2[i] = sol.y_events[1][0][4]
        x3[i] = sol.y_events[2][0][6]
        y3[i] = sol.y_events[2][0][7]
          
def save_data(file):
    '''Save X, Y, and TOF data to a binary file.'''
    delay = np.zeros(num) #zero array placeholders
    adc1 = np.zeros(num)
    adc2 = np.zeros(num)
    index = np.zeros(num)
    xyt_all = np.zeros((num, 13)) #array to save as binary
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
    xyt_all[:,10] = adc1
    xyt_all[:,11] = adc2
    xyt_all[:,12] = index
    np.save(file, xyt_all) #save array as a binary file
        
sol = simulate(r11, r21, r22, r32) #run simulator with given initial positions
save_data(file)        #save the data
