# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:42:42 2020
Nathan Marshall

"""
import numpy as np
from random import gauss
from scipy.integrate import solve_ivp
from numba import jit
import matplotlib.pyplot as plt
from ColtrimsAnalysis import hist1d, hist2d

class IncompleteThreeBody:
    
    def __init__(self, m1, m2, m3, q1, q2, q3, r10, 
                 r20, r30, t0, tmax, V, L, num, vibmax):
        
        r10 = np.asarray(r10)
        r20 = np.asarray(r20)
        r30 = np.asarray(r30)
        
        k = 8.9875517923e9 #Coulomb force constant
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        m1 = m1 * 1.66053903e-27 #convert mass from g/mole to kg
        m2 = m2 * 1.66053903e-27
        m3 = m3 * 1.66053903e-27 
        q1 = q1 * 1.60217662e-19
        q2 = q2 * 1.60217662e-19
        q3 = q3 * 1.60217662e-19
     
        self.tof1 = np.zeros(num) #create arrays to store output data
        self.x1 = np.zeros(num)
        self.y1 = np.zeros(num)
        self.tof2 = np.zeros(num)
        self.x2 = np.zeros(num)
        self.y2 = np.zeros(num)
        
        
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
            dvx1, dvy1, dvz1 = coul(r1, q1, m1, r2, q2) + coul(r1, q1, m1, r3, q3) + spec(q1, m1) 
            dvx2, dvy2, dvz2 = coul(r2, q2, m2, r1, q1) + coul(r2, q2, m2, r3, q3) + spec(q2, m2)
            dvx3, dvy3, dvz3 = coul(r3, q3, m3, r1, q1) + coul(r3, q3, m3, r2, q2) + spec(q3, m3)
            
            return(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, 
                   dvx1, dvy1, dvz1, dvx2, dvy2, dvz2, dvx3, dvy3, dvz3)
        
        def hit1(t, d): return L - d[2] #functions for detecting spectrometer hits
          
        def hit2(t, d): return L - d[5]
              
        for i in range(num):
            
            #add a random vibration to initial position vectors 
            r10_vib = r10 + rand_vector() * vibmax * np.random.rand()
            r20_vib = r20 + rand_vector() * vibmax * np.random.rand()
            r30_vib = r30 + rand_vector() * vibmax * np.random.rand()
            
            axis = rand_vector() #choose random axis
            theta = 2 * np.pi * np.random.rand() #choose random angle
            
            #rotate the position vectors around random axis by random angle
            r10_vib = np.dot(rotation_matrix(axis, theta), r10_vib)
            r20_vib = np.dot(rotation_matrix(axis, theta), r20_vib)
            r30_vib = np.dot(rotation_matrix(axis, theta), r30_vib)
            
            x10, y10, z10 = r10_vib #unpack fragment initial position vectors
            x20, y20, z20 = r20_vib
            x30, y30, z30 = r30_vib
            
            #define initial conditions list for the diffeq solver
            ivs = [x10, y10, z10, x20, y20, z20, x30, 
                   y30, z30, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            #run differential equation solver with initial values
            sol = solve_ivp(diffeq, [t0, tmax], ivs, events=(hit1, hit2))
           
            #check for true detector hits and extract tof, x, and y
            if sol.t_events[0].size !=0 and sol.t_events[1].size !=0:
                self.tof1[i] = sol.t_events[0][0]
                self.tof2[i] = sol.t_events[1][0]
                self.x1[i] = sol.y_events[0][0][0]
                self.y1[i] = sol.y_events[0][0][1]
                self.x2[i] = sol.y_events[1][0][3]
                self.y2[i] = sol.y_events[1][0][4]
                
        self.tof1 = self.tof1 * 1e9
        self.x1 = self.x1 * 1000
        self.y1 = self.y1 * 1000
        self.tof2 = self.tof2 * 1e9
        self.x2 = self.x2 * 1000
        self.y2 = self.y2 * 1000
        
    def p_ke_2body(self, param_list, ion_form):
       
        da_to_au = 1822.8885 #conversion factor from daltons to atomic units
        mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
        au_to_ev = 27.211386 #conv. factor from a.u. energy to eV
        
        self.param_list = param_list
        self.ion1, self.ion2 = ion_form
        self.ion_form = ion_form
        l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
        m1, m2 = [da_to_au * i for i in [self.m1, self.m2]]
        acc1 = (2 * self.q1 * (l - z0)) / (m1 * C**2) #acceleration of 1st ion
        acc2 = (2 * self.q2 * (l - z0)) / (m2 * C**2) #acceleration of 2nd ion
        self.vx1 = ((self.x1 - (x_jet))/(self.tof1 - t0)) - (vx_jet)  
        self.vy1 = ((self.y1 - (y_jet))/(self.tof1 - t0)) - (vy_jet)
        self.vz1 = (l-z0)/(self.tof1-t0) - (1/2)*acc1*(self.tof1-t0) + vz0
        self.vx2 = ((self.x2 - (x_jet))/(self.tof2 - t0)) - (vx_jet)
        self.vy2 = ((self.y2 - (y_jet))/(self.tof2 - t0)) - (vy_jet)
        self.vz2 = (l-z0)/(self.tof2-t0) - (1/2)*acc2*(self.tof2-t0) + vz0
        self.px1 = m1 * self.vx1 * mm_ns_to_au
        self.py1 = m1 * self.vy1 * mm_ns_to_au
        self.pz1 = m1 * self.vz1 * mm_ns_to_au
        self.px2 = m2 * self.vx2 * mm_ns_to_au
        self.py2 = m2 * self.vy2 * mm_ns_to_au
        self.pz2 = m2 * self.vz2 * mm_ns_to_au
        self.p_ion1 = [self.px1, self.py1, self.pz1]
        self.p_ion2 = [self.px2, self.py2, self.pz2]
        self.ptotx = self.px1 + self.px2 
        self.ptoty = self.py1 + self.py2 
        self.ptotz = self.pz1 + self.pz2 
        self.kex1 = self.px1**2 / (2 * m1) * au_to_ev
        self.kex2 = self.px2**2 / (2 * m2) * au_to_ev
        self.key1 = self.py1**2 / (2 * m1) * au_to_ev
        self.key2 = self.py2**2 / (2 * m2) * au_to_ev
        self.kez1 = self.pz1**2 / (2 * m1) * au_to_ev
        self.kez2 = self.pz2**2 / (2 * m2) * au_to_ev
        self.ke_tot1 = self.kex1 + self.key1 + self.kez1
        self.ke_tot2 = self.kex2 + self.key2 + self.kez2
        self.ker = self.ke_tot1 + self.ke_tot2
        
    def cos_ker(self):
            cos = self.pz1/np.sqrt(self.px1**2 + self.py1**2 + self.pz1**2)
            plt.style.use('dark_background')
            fig, ax = plt.subplots(1, 1)
            fig.canvas.set_window_title('Cos(theta) vs. KER')
            hist2d(self.ker, cos, ax, r'{}, {} Cos($\theta$) vs. KER'.format(
                   self.ion1, self.ion2), 'Kinetic Energy Release (eV)', 
                   r'Cos($\theta$)')
            
    def plot_pxyz(self):
        plt.style.use('default')
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
        fig.suptitle('{} Momentum'.format(self.ion1))
        fig.canvas.set_window_title('Ion 1 XYZ Momentum')
        hist1d(self.px1, ax1)
        hist1d(-(self.px1), ax1, 'X Momentum', 'Momentum (a.u.)', 'Counts')
        ax1.legend(['$P_x$', '$-P_x$'], loc=1)
        hist1d(self.py1, ax2)
        hist1d(-(self.py1), ax2, 'Y Momentum', 'Momentum (a.u.)')
        ax2.legend(['$P_y$', '$-P_y$'], loc=1)
        hist1d(self.pz1, ax3)
        hist1d(-(self.pz1), ax3, 'Z Momentum', 'Momentum (a.u.)')
        ax3.legend(['$P_z$', '$-P_z$'], loc=1)
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
        fig.suptitle('{} Momentum'.format(self.ion2))
        fig.canvas.set_window_title('Ion 2 XYZ Momentum')
        hist1d(self.px2, ax1)
        hist1d(-(self.px2), ax1, 'X Momentum', 'Momentum (a.u.)', 'Counts')
        ax1.legend(['$P_x$', '$-P_x$'], loc=1)
        hist1d(self.py2, ax2)
        hist1d(-(self.py2), ax2, 'Y Momentum', 'Momentum (a.u.)')
        ax2.legend(['$P_y$', '$-P_y$'], loc=1)
        hist1d(self.pz2, ax3)
        hist1d(-(self.pz2), ax3, 'Z Momentum', 'Momentum (a.u.)')
        ax3.legend(['$P_z$', '$-P_z$'], loc=1)
            
    def plot_psum(self, binsize='default'):
        plt.style.use('default')
        fig, ax = plt.subplots(1, 3)
        fig.canvas.set_window_title('Momentum Sums')
        hist1d(self.ptotx, ax[0], 'X Momentum Sum','X Momentum (a.u.)', 
               'Counts', binsize=binsize)
        hist1d(self.ptoty, ax[1], 'Y Momentum Sum','Y Momentum (a.u.)', '', 
               binsize=binsize)
        hist1d(self.ptotz, ax[2], 'Z Momentum Sum','Z Momentum (a.u.)', '', 
               binsize=binsize)
        title = self.ion1 + ' and ' + self.ion2 + ' Momentum Sums'
        fig.suptitle(title)
        
    def plot_energy(self, binsize='default'):
        fig, ax = plt.subplots(1, 3)
        fig.canvas.set_window_title('Ion Kinetic Energy and KER')
        hist1d(self.ke_tot1, ax[0],'{} Kinetic Energy'.format(self.ion1),
               'Kinetic Energy (eV)','Counts', binsize=binsize)
        hist1d(self.ke_tot2, ax[1],'{} Kinetic Energy'.format(self.ion2), 
               'Kinetic Energy (eV)', '', binsize=binsize)
        hist1d(self.ker, ax[2], 'Kinetic Energy Release',
               'Kinetic Energy (eV)', '', binsize=binsize)
        
    def save_data(self, file):
        '''Save X, Y, and TOF data to a binary file.'''
        delay = np.zeros(self.num) #zero array placeholders
        adc1 = np.zeros(self.num)
        adc2 = np.zeros(self.num)
        index = np.zeros(self.num)
        xyt_all = np.zeros((self.num, 10)) #array to save as binary
        xyt_all[:,0] = delay
        xyt_all[:,1] = self.x1 
        xyt_all[:,2] = self.y1 
        xyt_all[:,3] = self.tof1 
        xyt_all[:,4] = self.x2 
        xyt_all[:,5] = self.y2 
        xyt_all[:,6] = self.tof2
        xyt_all[:,7] = adc1
        xyt_all[:,8] = adc2
        xyt_all[:,9] = index
        np.save(file, xyt_all) #save array as a binary file