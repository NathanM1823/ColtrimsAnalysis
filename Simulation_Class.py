# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:51:31 2020
Nathan Marshall

"""
import numpy as np
from random import gauss
from scipy.integrate import solve_ivp
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from ColtrimsAnalysis import hist1d, hist2d

class ConcertedThreeBody:
    
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
        self.tof3 = np.zeros(num)
        self.x3 = np.zeros(num)
        self.y3 = np.zeros(num)
        
        
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
        
        def hit3(t, d): return L - d[8]
        

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
            sol = solve_ivp(diffeq, [t0, tmax], ivs, events=(hit1, hit2, hit3))
           
            #check for true detector hits and extract tof, x, and y
            if sol.t_events[0].size !=0 and sol.t_events[1].size !=0 and sol.t_events[2].size != 0:
                self.tof1[i] = sol.t_events[0][0]
                self.tof2[i] = sol.t_events[1][0]
                self.tof3[i] = sol.t_events[2][0]
                self.x1[i] = sol.y_events[0][0][0]
                self.y1[i] = sol.y_events[0][0][1]
                self.x2[i] = sol.y_events[1][0][3]
                self.y2[i] = sol.y_events[1][0][4]
                self.x3[i] = sol.y_events[2][0][6]
                self.y3[i] = sol.y_events[2][0][7]
                
        self.tof1 = self.tof1 * 1e9
        self.x1 = self.x1 * 1000
        self.y1 = self.y1 * 1000
        self.tof2 = self.tof2 * 1e9
        self.x2 = self.x2 * 1000
        self.y2 = self.y2 * 1000
        self.tof3 = self.tof3 * 1e9
        self.x3 = self.x3 * 1000
        self.y3 = self.y3 * 1000
        
    def p_ke_3body(self, param_list, ion_form):
       
        da_to_au = 1822.8885 #conversion factor from daltons to atomic units
        mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
        au_to_ev = 27.211386 #conv. factor from a.u. energy to eV
        
        self.param_list = param_list
        self.ion1, self.ion2, self.ion3 = ion_form
        self.ion_form = ion_form
        l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
        m1, m2, m3 = [da_to_au * i for i in [self.m1, self.m2, self.m3]]
        acc1 = (2 * self.q1 * (l - z0)) / (m1 * C**2) #acceleration of 1st ion
        acc2 = (2 * self.q2 * (l - z0)) / (m2 * C**2) #acceleration of 2nd ion
        acc3 = (2 * self.q3 * (l - z0)) / (m3 * C**2) #acceleration of 3rd ion
        self.vx1 = ((self.x1 - (x_jet))/(self.tof1 - t0)) - (vx_jet)  
        self.vy1 = ((self.y1 - (y_jet))/(self.tof1 - t0)) - (vy_jet)
        self.vz1 = (l-z0)/(self.tof1-t0) - (1/2)*acc1*(self.tof1-t0) + vz0
        self.vx2 = ((self.x2 - (x_jet))/(self.tof2 - t0)) - (vx_jet)
        self.vy2 = ((self.y2 - (y_jet))/(self.tof2 - t0)) - (vy_jet)
        self.vz2 = (l-z0)/(self.tof2-t0) - (1/2)*acc2*(self.tof2-t0) + vz0
        self.vx3 = ((self.x3 - (x_jet))/(self.tof3 - t0)) - (vx_jet)
        self.vy3 = ((self.y3 - (y_jet))/(self.tof3 - t0)) - (vy_jet)
        self.vz3 = (l-z0)/(self.tof3-t0) - (1/2)*acc3*(self.tof3-t0) + vz0
        self.px1 = m1 * self.vx1 * mm_ns_to_au
        self.py1 = m1 * self.vy1 * mm_ns_to_au
        self.pz1 = m1 * self.vz1 * mm_ns_to_au
        self.px2 = m2 * self.vx2 * mm_ns_to_au
        self.py2 = m2 * self.vy2 * mm_ns_to_au
        self.pz2 = m2 * self.vz2 * mm_ns_to_au
        self.px3 = m3 * self.vx3 * mm_ns_to_au
        self.py3 = m3 * self.vy3 * mm_ns_to_au
        self.pz3 = m3 * self.vz3 * mm_ns_to_au
        self.p_ion1 = [self.px1, self.py1, self.pz1]
        self.p_ion2 = [self.px2, self.py2, self.pz2]
        self.p_ion3 = [self.px3, self.py3, self.pz3]
        self.ptotx = self.px1 + self.px2 + self.px3
        self.ptoty = self.py1 + self.py2 + self.py3
        self.ptotz = self.pz1 + self.pz2 + self.pz3
        self.kex1 = self.px1**2 / (2 * m1) * au_to_ev
        self.kex2 = self.px2**2 / (2 * m2) * au_to_ev
        self.kex3 = self.px3**2 / (2 * m3) * au_to_ev
        self.key1 = self.py1**2 / (2 * m1) * au_to_ev
        self.key2 = self.py2**2 / (2 * m2) * au_to_ev
        self.key3 = self.py3**2 / (2 * m3) * au_to_ev
        self.kez1 = self.pz1**2 / (2 * m1) * au_to_ev
        self.kez2 = self.pz2**2 / (2 * m2) * au_to_ev
        self.kez3 = self.pz3**2 / (2 * m3) * au_to_ev
        self.ke_tot1 = self.kex1 + self.key1 + self.kez1
        self.ke_tot2 = self.kex2 + self.key2 + self.kez2
        self.ke_tot3 = self.kex3 + self.key3 + self.kez3
        self.ker = self.ke_tot1 + self.ke_tot2 + self.ke_tot3
        
    def newton_plots(self, xbin='default', ybin='default'):
        '''
        Generates 3 Newton Plots. In each plot, one of ion's momentum is set as 
        a unit vector on the x-axis, and the momenta of the other two ions are 
        plotted relative to it.
        '''
        def newtoncalc(plot_input, xbin, ybin):
            p_ion1, p_ion2, p_ion3, ion1, ion2, ion3 = plot_input
            px1, py1, pz1 = p_ion1
            px2, py2, pz2 = p_ion2
            px3, py3, pz3 = p_ion3
            dot1_2 = px1*px2 + py1*py2 + pz1*pz2
            dot1_3 = px1*px3 + py1*py3 + pz1*pz3
            pmag1 = np.sqrt(px1**2 + py1**2 + pz1**2)
            pmag2 = np.sqrt(px2**2 + py2**2 + pz2**2)
            pmag3 = np.sqrt(px3**2 + py3**2 + pz3**2)
            px2_newton = dot1_2/pmag1
            py2_newton = np.sqrt(pmag2**2 - px2_newton**2)
            px3_newton = dot1_3/pmag1
            py3_newton = -np.sqrt(pmag3**2 - px3_newton**2)
            px_newton = np.concatenate((px2_newton/pmag1, px3_newton/pmag1))
            py_newton = np.concatenate((py2_newton/pmag1, py3_newton/pmag1))
            plt.style.use('default')
            fig, ax = plt.subplots(1, 1)
            title = 'Newton Plot Relative to {}'.format(ion1)
            fig.canvas.set_window_title(title)
            hist2d(px_newton, py_newton, ax, title, 'Relative X Momentum', 
                   'Relative Y Momentum', xbinsize=xbin, ybinsize=ybin, 
                   color_map='viridis')
            ax.quiver(1, 0, color='r', scale=1, scale_units='x', headlength=4,
                      headaxislength=4)
            ax.axhline(y=0, color='black', linewidth=0.8)
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.text(1.02, 0.08, ion1, fontsize=12)
            ax.text(0.01, 0.93, ion2, fontsize=12, transform=ax.transAxes)
            ax.text(0.01, 0.03, ion3, fontsize=12, transform=ax.transAxes)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5 ,5)
        plot1 = [self.p_ion1, self.p_ion2, self.p_ion3, self.ion1, self.ion2,
                 self.ion3]
        plot2 = [self.p_ion2, self.p_ion3, self.p_ion1, self.ion2, self.ion3,
                 self.ion1]
        plot3 = [self.p_ion3, self.p_ion1, self.p_ion2, self.ion3, self.ion1,
                 self.ion2]
        newtoncalc(plot1, xbin, ybin)
        newtoncalc(plot2, xbin, ybin)
        newtoncalc(plot3, xbin, ybin)
    
    def dalitz1(self, xbin, ybin):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_data = (epsilon2 - epsilon1) / (3**(1/2))
        y_data = epsilon3 - 1/3
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Dalitz Plot 1')
        xlabel = r'$(\epsilon_2 - \epsilon_1)/\sqrt{3} $'
        ylabel = r'$\epsilon_3 - \frac{1}{3}$'
        hist2d(x_data, y_data, ax, 'Dalitz Plot', xlabel, ylabel, 
               xbinsize=xbin, ybinsize=ybin, color_map='viridis')
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.4, 0.6)
        ax.set_aspect('equal')
        circle = mpatches.Circle((0,0), 1/3, fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        plt.tight_layout()
    
    def dalitz2(self, xbin, ybin):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_data = (epsilon1 - epsilon3) / (3**(1/2))
        y_data = epsilon2 - 1/3
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Dalitz Plot 2')
        xlabel = r'$(\epsilon_1 - \epsilon_3)/\sqrt{3} $'
        ylabel = r'$\epsilon_2 - \frac{1}{3}$'
        hist2d(x_data, y_data, ax, 'Dalitz Plot', xlabel, ylabel, 
               xbinsize=xbin, ybinsize=ybin, color_map='viridis')
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.4, 0.6)
        ax.set_aspect('equal')
        circle = mpatches.Circle((0,0), 1/3, fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        plt.tight_layout()
        
    def dalitz3(self, xbin, ybin):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_data = (epsilon2 - epsilon3) / (3**(1/2))
        y_data = epsilon1 - 1/3
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Dalitz Plot 3')
        xlabel = r'$(\epsilon_2 - \epsilon_3)/\sqrt{3} $'
        ylabel = r'$\epsilon_1 - \frac{1}{3}$'
        hist2d(x_data, y_data, ax, 'Dalitz Plot', xlabel, ylabel, 
               xbinsize=xbin, ybinsize=ybin, color_map='viridis')
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.4, 0.6)
        ax.set_aspect('equal')
        circle = mpatches.Circle((0,0), 1/3, fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        plt.tight_layout()
    
    def mom_sphere(self, xbin, ybin):
        plt.style.use('dark_background')
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
        fig.suptitle('Momentum Spheres {}'.format(self.ion1))
        fig.canvas.set_window_title('Momentum Spheres')
        hist2d(self.px1, self.py1, ax1, '$P_x$ vs. $P_y$', '$P_x$', '$P_y$',
               xbinsize=xbin, ybinsize=ybin, colorbar=False)
        hist2d(self.py1, self.pz1, ax2, '$P_y$ vs. $P_z$', '$P_y$', '$P_z$',
               xbinsize=xbin, ybinsize=ybin, colorbar=False)
        hist2d(self.pz1, self.px1, ax3, '$P_z$ vs. $P_x$', '$P_z$', '$P_x$',
               xbinsize=xbin, ybinsize=ybin, )
        
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
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
        fig.suptitle('{} Momentum'.format(self.ion3))
        fig.canvas.set_window_title('Ion 3 XYZ Momentum')
        hist1d(self.px3, ax1)
        hist1d(-(self.px3), ax1, 'X Momentum', 'Momentum (a.u.)', 'Counts')
        ax1.legend(['$P_x$', '$-P_x$'], loc=1)
        hist1d(self.py3, ax2)
        hist1d(-(self.py3), ax2, 'Y Momentum', 'Momentum (a.u.)')
        ax2.legend(['$P_y$', '$-P_y$'], loc=1)
        hist1d(self.pz3, ax3)
        hist1d(-(self.pz3), ax3, 'Z Momentum', 'Momentum (a.u.)')
        ax3.legend(['$P_z$', '$-P_z$'], loc=1)
    
    def plot_psum(self, binsize='default'):
        plt.style.use('default')
        fig, ax = plt.subplots(1, 3)
        fig.canvas.set_window_title('Momentum Sums')
        hist1d(self.ptotx, ax[0], 'X Momentum Sum', 'X Momentum (a.u.)', 
               'Counts', binsize=binsize)
        hist1d(self.ptoty, ax[1], 'Y Momentum Sum','Y Momentum (a.u.)', '', 
               binsize=binsize)
        hist1d(self.ptotz, ax[2], 'Z Momentum Sum','Z Momentum (a.u.)', '', 
               binsize=binsize)
        title = (self.ion1 + ', ' + self.ion2 + ', ' 
                 + self.ion3 + ' Momentum Sums')
        fig.suptitle(title)
        
    def plot_energy(self, binsize='default'):
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Ion Kinetic Energy and Total KER')
        hist1d(self.ke_tot1, ax, '', '', '', binsize=binsize)
        hist1d(self.ke_tot2, ax, '', '', '', binsize=binsize)
        hist1d(self.ke_tot3, ax, '', '', '', binsize=binsize)
        hist1d(self.ker, ax, '{} , {}, {} Kinetic Energy'.format(self.ion1, 
               self.ion2, self.ion3), 'Kinetic Energy (eV)', 'Counts', 
               binsize=binsize)
        ax.legend([self.ion1, self.ion2, self.ion3, 'Total KER'])
        
           
    def save_data(self, file):
        '''Save X, Y, and TOF data to a binary file.'''
        delay = np.zeros(self.num) #zero array placeholders
        adc1 = np.zeros(self.num)
        adc2 = np.zeros(self.num)
        index = np.zeros(self.num)
        xyt_all = np.zeros((self.num, 13)) #array to save as binary
        xyt_all[:,0] = delay
        xyt_all[:,1] = self.x1 
        xyt_all[:,2] = self.y1 
        xyt_all[:,3] = self.tof1 
        xyt_all[:,4] = self.x2 
        xyt_all[:,5] = self.y2 
        xyt_all[:,6] = self.tof2
        xyt_all[:,7] = self.x3 
        xyt_all[:,8] = self.y3
        xyt_all[:,9] = self.tof3
        xyt_all[:,10] = adc1
        xyt_all[:,11] = adc2
        xyt_all[:,12] = index
        np.save(file, xyt_all) #save array as a binary file
        

class SequentialThreeBody:
    def __init__(self, m11, m21, m12, m22, m32, q11, q21, q12, q22, q32, 
                 r11, r21, r12, r22, t0, t_frag, tmax, V, L, num, vibmax, max_angle):
        
        r11 = np.asarray(r11)
        r21 = np.asarray(r21)
        r12 = np.asarray(r12)
        r22 = np.asarray(r22)
        
        r12 = r12 - r11
        r22 = r22 - r11
        
        self.m1 = m12
        self.m2 = m22
        self.m3 = m32
        self.q1 = q12
        self.q2 = q22
        self.q3 = q32
        self.num = num
        
        m11 = m11 * 1.66053903e-27 #convert mass from g/mole to kg
        m21 = m21 * 1.66053903e-27
        m12 = m12 * 1.66053903e-27
        m22 = m22 * 1.66053903e-27
        m32 = m32 * 1.66053903e-27
        
        q11 = q11 * 1.60217662e-19 #fragment charges before sequential fragmentation
        q21 = q21 * 1.60217662e-19
        
        q12 = q12 * 1.60217662e-19 #fragment charges after sequential fragmentation
        q22 = q22 * 1.60217662e-19
        q32 = q32 * 1.60217662e-19
        
        k = 8.9875517923e9 #Coulomb force constant
        
        self.tof1 = np.zeros(num) #create arrays to store output data
        self.x1 = np.zeros(num)
        self.y1 = np.zeros(num)
        self.tof2 = np.zeros(num)
        self.x2 = np.zeros(num)
        self.y2 = np.zeros(num)
        self.tof3 = np.zeros(num)
        self.x3 = np.zeros(num)
        self.y3 = np.zeros(num)
        
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
        
        
        for i in range(num):
            
            #add a random vibration to initial position vectors 
            r11_vib = r11 + rand_vector() * vibmax * np.random.rand()
            r21_vib = r21 + rand_vector() * vibmax * np.random.rand()
            
            axis = rand_vector() #choose random axis
            theta = 2 * np.pi * np.random.rand() #choose random angle
            
            #rotate the position vectors around random axis by random angle
            r11_vib = np.dot(rotation_matrix(axis, theta), r11_vib)
            r21_vib = np.dot(rotation_matrix(axis, theta), r21_vib)
            r12_rot = np.dot(rotation_matrix(axis, theta), r12)
            r22_rot = np.dot(rotation_matrix(axis, theta), r22)
            
            x11, y11, z11 = r11_vib #unpack fragment initial position vectors
            x21, y21, z21 = r21_vib
            
            #define initial conditions list for the diffeq solver
            ivs = [x11, y11, z11, x21, y21, z21, 0, 0, 0, 0, 0, 0]
            
            #run differential equation solver with initial values
            sol = solve_ivp(diffeq1, [t0, t_frag], ivs)
            
            r11_out = np.array([sol.y[0][-1], sol.y[1][-1], sol.y[2][-1]])
            r21_out = np.array([sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]])
            v11 = np.array([sol.y[6][-1], sol.y[7][-1], sol.y[8][-1]])
            v21 = np.array([sol.y[9][-1], sol.y[10][-1], sol.y[11][-1]])
            
            axis = rand_vector() #choose random axis
            theta = max_angle * np.random.rand() #choose random angle
            
            r12_rot = np.dot(rotation_matrix(axis, theta), r12_rot)
            r22_rot = np.dot(rotation_matrix(axis, theta), r22_rot)
            
            r12_mod = r11_out + r12_rot
            r22_mod = r11_out + r22_rot
            r32_mod = r21_out
        
            x12, y12, z12 = r12_mod #unpack fragment initial position vectors
            x22, y22, z22 = r22_mod
            x32, y32, z32 = r32_mod
            
            vx11, vy11, vz11 = v11
            vx21, vy21, vz21 = v21
            
            #define initial conditions list for the diffeq solver
            ivs = [x12, y12, z12, x22, y22, z22, x32, y32, z32, 
                   vx11, vy11, vz11, vx11, vy11, vz11, vx21, vy21, vz21]
                
            #run differential equation solver with initial values
            sol = solve_ivp(diffeq2, [t_frag, tmax], ivs, events=(hit1, hit2, hit3))
            
            #check for true detector hits and extract tof, x, and y
            if (sol.t_events[0].size != 0 and sol.t_events[1].size != 0 
                and sol.t_events[2].size != 0):
                self.tof1[i] = sol.t_events[0][0]
                self.tof2[i] = sol.t_events[1][0]
                self.tof3[i] = sol.t_events[2][0]
                self.x1[i] = sol.y_events[0][0][0]
                self.y1[i] = sol.y_events[0][0][1]
                self.x2[i] = sol.y_events[1][0][3]
                self.y2[i] = sol.y_events[1][0][4]
                self.x3[i] = sol.y_events[2][0][6]
                self.y3[i] = sol.y_events[2][0][7]
                    
        self.tof1 = self.tof1 * 1e9
        self.x1 = self.x1 * 1000
        self.y1 = self.y1 * 1000
        self.tof2 = self.tof2 * 1e9
        self.x2 = self.x2 * 1000
        self.y2 = self.y2 * 1000
        self.tof3 = self.tof3 * 1e9
        self.x3 = self.x3 * 1000
        self.y3 = self.y3 * 1000
        
    def p_ke_3body(self, param_list, ion_form):
       
        da_to_au = 1822.8885 #conversion factor from daltons to atomic units
        mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
        au_to_ev = 27.211386 #conv. factor from a.u. energy to eV
        
        self.param_list = param_list
        self.ion1, self.ion2, self.ion3 = ion_form
        self.ion_form = ion_form
        l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
        m1, m2, m3 = [da_to_au * i for i in [self.m1, self.m2, self.m3]]
        acc1 = (2 * self.q1 * (l - z0)) / (m1 * C**2) #acceleration of 1st ion
        acc2 = (2 * self.q2 * (l - z0)) / (m2 * C**2) #acceleration of 2nd ion
        acc3 = (2 * self.q3 * (l - z0)) / (m3 * C**2) #acceleration of 3rd ion
        self.vx1 = ((self.x1 - (x_jet))/(self.tof1 - t0)) - (vx_jet)  
        self.vy1 = ((self.y1 - (y_jet))/(self.tof1 - t0)) - (vy_jet)
        self.vz1 = (l-z0)/(self.tof1-t0) - (1/2)*acc1*(self.tof1-t0) + vz0
        self.vx2 = ((self.x2 - (x_jet))/(self.tof2 - t0)) - (vx_jet)
        self.vy2 = ((self.y2 - (y_jet))/(self.tof2 - t0)) - (vy_jet)
        self.vz2 = (l-z0)/(self.tof2-t0) - (1/2)*acc2*(self.tof2-t0) + vz0
        self.vx3 = ((self.x3 - (x_jet))/(self.tof3 - t0)) - (vx_jet)
        self.vy3 = ((self.y3 - (y_jet))/(self.tof3 - t0)) - (vy_jet)
        self.vz3 = (l-z0)/(self.tof3-t0) - (1/2)*acc3*(self.tof3-t0) + vz0
        self.px1 = m1 * self.vx1 * mm_ns_to_au
        self.py1 = m1 * self.vy1 * mm_ns_to_au
        self.pz1 = m1 * self.vz1 * mm_ns_to_au
        self.px2 = m2 * self.vx2 * mm_ns_to_au
        self.py2 = m2 * self.vy2 * mm_ns_to_au
        self.pz2 = m2 * self.vz2 * mm_ns_to_au
        self.px3 = m3 * self.vx3 * mm_ns_to_au
        self.py3 = m3 * self.vy3 * mm_ns_to_au
        self.pz3 = m3 * self.vz3 * mm_ns_to_au
        self.p_ion1 = [self.px1, self.py1, self.pz1]
        self.p_ion2 = [self.px2, self.py2, self.pz2]
        self.p_ion3 = [self.px3, self.py3, self.pz3]
        self.ptotx = self.px1 + self.px2 + self.px3
        self.ptoty = self.py1 + self.py2 + self.py3
        self.ptotz = self.pz1 + self.pz2 + self.pz3
        self.kex1 = self.px1**2 / (2 * m1) * au_to_ev
        self.kex2 = self.px2**2 / (2 * m2) * au_to_ev
        self.kex3 = self.px3**2 / (2 * m3) * au_to_ev
        self.key1 = self.py1**2 / (2 * m1) * au_to_ev
        self.key2 = self.py2**2 / (2 * m2) * au_to_ev
        self.key3 = self.py3**2 / (2 * m3) * au_to_ev
        self.kez1 = self.pz1**2 / (2 * m1) * au_to_ev
        self.kez2 = self.pz2**2 / (2 * m2) * au_to_ev
        self.kez3 = self.pz3**2 / (2 * m3) * au_to_ev
        self.ke_tot1 = self.kex1 + self.key1 + self.kez1
        self.ke_tot2 = self.kex2 + self.key2 + self.kez2
        self.ke_tot3 = self.kex3 + self.key3 + self.kez3
        self.ker = self.ke_tot1 + self.ke_tot2 + self.ke_tot3
        
    def newton_plots(self, xbin='default', ybin='default'):
        '''
        Generates 3 Newton Plots. In each plot, one of ion's momentum is set as 
        a unit vector on the x-axis, and the momenta of the other two ions are 
        plotted relative to it.
        '''
        def newtoncalc(plot_input, xbin, ybin):
            p_ion1, p_ion2, p_ion3, ion1, ion2, ion3 = plot_input
            px1, py1, pz1 = p_ion1
            px2, py2, pz2 = p_ion2
            px3, py3, pz3 = p_ion3
            dot1_2 = px1*px2 + py1*py2 + pz1*pz2
            dot1_3 = px1*px3 + py1*py3 + pz1*pz3
            pmag1 = np.sqrt(px1**2 + py1**2 + pz1**2)
            pmag2 = np.sqrt(px2**2 + py2**2 + pz2**2)
            pmag3 = np.sqrt(px3**2 + py3**2 + pz3**2)
            px2_newton = dot1_2/pmag1
            py2_newton = np.sqrt(pmag2**2 - px2_newton**2)
            px3_newton = dot1_3/pmag1
            py3_newton = -np.sqrt(pmag3**2 - px3_newton**2)
            px_newton = np.concatenate((px2_newton/pmag1, px3_newton/pmag1))
            py_newton = np.concatenate((py2_newton/pmag1, py3_newton/pmag1))
            plt.style.use('default')
            fig, ax = plt.subplots(1, 1)
            title = 'Newton Plot Relative to {}'.format(ion1)
            fig.canvas.set_window_title(title)
            hist2d(px_newton, py_newton, ax, title, 'Relative X Momentum', 
                   'Relative Y Momentum', xbinsize=xbin, ybinsize=ybin, 
                   color_map='viridis')
            ax.quiver(1, 0, color='r', scale=1, scale_units='x', headlength=4,
                      headaxislength=4)
            ax.axhline(y=0, color='black', linewidth=0.8)
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.text(1.02, 0.08, ion1, fontsize=12)
            ax.text(0.01, 0.93, ion2, fontsize=12, transform=ax.transAxes)
            ax.text(0.01, 0.03, ion3, fontsize=12, transform=ax.transAxes)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5 ,5)
        plot1 = [self.p_ion1, self.p_ion2, self.p_ion3, self.ion1, self.ion2,
                 self.ion3]
        plot2 = [self.p_ion2, self.p_ion3, self.p_ion1, self.ion2, self.ion3,
                 self.ion1]
        plot3 = [self.p_ion3, self.p_ion1, self.p_ion2, self.ion3, self.ion1,
                 self.ion2]
        newtoncalc(plot1, xbin, ybin)
        newtoncalc(plot2, xbin, ybin)
        newtoncalc(plot3, xbin, ybin)
    
    def dalitz1(self, xbin, ybin):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_data = (epsilon2 - epsilon1) / (3**(1/2))
        y_data = epsilon3 - 1/3
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Dalitz Plot 1')
        xlabel = r'$(\epsilon_2 - \epsilon_1)/\sqrt{3} $'
        ylabel = r'$\epsilon_3 - \frac{1}{3}$'
        hist2d(x_data, y_data, ax, 'Dalitz Plot', xlabel, ylabel, 
               xbinsize=xbin, ybinsize=ybin, color_map='viridis')
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.4, 0.6)
        ax.set_aspect('equal')
        circle = mpatches.Circle((0,0), 1/3, fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        plt.tight_layout()
    
    def dalitz2(self, xbin, ybin):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_data = (epsilon1 - epsilon3) / (3**(1/2))
        y_data = epsilon2 - 1/3
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Dalitz Plot 2')
        xlabel = r'$(\epsilon_1 - \epsilon_3)/\sqrt{3} $'
        ylabel = r'$\epsilon_2 - \frac{1}{3}$'
        hist2d(x_data, y_data, ax, 'Dalitz Plot', xlabel, ylabel, 
               xbinsize=xbin, ybinsize=ybin, color_map='viridis')
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.4, 0.6)
        ax.set_aspect('equal')
        circle = mpatches.Circle((0,0), 1/3, fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        plt.tight_layout()
        
    def dalitz3(self, xbin, ybin):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_data = (epsilon2 - epsilon3) / (3**(1/2))
        y_data = epsilon1 - 1/3
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Dalitz Plot 3')
        xlabel = r'$(\epsilon_2 - \epsilon_3)/\sqrt{3} $'
        ylabel = r'$\epsilon_1 - \frac{1}{3}$'
        hist2d(x_data, y_data, ax, 'Dalitz Plot', xlabel, ylabel, 
               xbinsize=xbin, ybinsize=ybin, color_map='viridis')
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.4, 0.6)
        ax.set_aspect('equal')
        circle = mpatches.Circle((0,0), 1/3, fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        plt.tight_layout()
    
    def mom_sphere(self, xbin, ybin):
        plt.style.use('dark_background')
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
        fig.suptitle('Momentum Spheres {}'.format(self.ion1))
        fig.canvas.set_window_title('Momentum Spheres')
        hist2d(self.px1, self.py1, ax1, '$P_x$ vs. $P_y$', '$P_x$', '$P_y$',
               xbinsize=xbin, ybinsize=ybin, colorbar=False)
        hist2d(self.py1, self.pz1, ax2, '$P_y$ vs. $P_z$', '$P_y$', '$P_z$',
               xbinsize=xbin, ybinsize=ybin, colorbar=False)
        hist2d(self.pz1, self.px1, ax3, '$P_z$ vs. $P_x$', '$P_z$', '$P_x$',
               xbinsize=xbin, ybinsize=ybin, )
        
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
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
        fig.suptitle('{} Momentum'.format(self.ion3))
        fig.canvas.set_window_title('Ion 3 XYZ Momentum')
        hist1d(self.px3, ax1)
        hist1d(-(self.px3), ax1, 'X Momentum', 'Momentum (a.u.)', 'Counts')
        ax1.legend(['$P_x$', '$-P_x$'], loc=1)
        hist1d(self.py3, ax2)
        hist1d(-(self.py3), ax2, 'Y Momentum', 'Momentum (a.u.)')
        ax2.legend(['$P_y$', '$-P_y$'], loc=1)
        hist1d(self.pz3, ax3)
        hist1d(-(self.pz3), ax3, 'Z Momentum', 'Momentum (a.u.)')
        ax3.legend(['$P_z$', '$-P_z$'], loc=1)
    
    def plot_psum(self, binsize='default'):
        plt.style.use('default')
        fig, ax = plt.subplots(1, 3)
        fig.canvas.set_window_title('Momentum Sums')
        hist1d(self.ptotx, ax[0], 'X Momentum Sum', 'X Momentum (a.u.)', 
               'Counts', binsize=binsize)
        hist1d(self.ptoty, ax[1], 'Y Momentum Sum','Y Momentum (a.u.)', '', 
               binsize=binsize)
        hist1d(self.ptotz, ax[2], 'Z Momentum Sum','Z Momentum (a.u.)', '', 
               binsize=binsize)
        title = (self.ion1 + ', ' + self.ion2 + ', ' 
                 + self.ion3 + ' Momentum Sums')
        fig.suptitle(title)
        
    def plot_energy(self, binsize='default'):
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Ion Kinetic Energy and Total KER')
        hist1d(self.ke_tot1, ax, '', '', '', binsize=binsize)
        hist1d(self.ke_tot2, ax, '', '', '', binsize=binsize)
        hist1d(self.ke_tot3, ax, '', '', '', binsize=binsize)
        hist1d(self.ker, ax, '{} , {}, {} Kinetic Energy'.format(self.ion1, 
               self.ion2, self.ion3), 'Kinetic Energy (eV)', 'Counts', 
               binsize=binsize)
        ax.legend([self.ion1, self.ion2, self.ion3, 'Total KER'])
          
    def save_data(self, file):
        '''Save X, Y, and TOF data to a binary file.'''
        delay = np.zeros(self.num) #zero array placeholders
        adc1 = np.zeros(self.num)
        adc2 = np.zeros(self.num)
        index = np.zeros(self.num)
        xyt_all = np.zeros((self.num, 13)) #array to save as binary
        xyt_all[:,0] = delay
        xyt_all[:,1] = self.x1 
        xyt_all[:,2] = self.y1 
        xyt_all[:,3] = self.tof1 
        xyt_all[:,4] = self.x2 
        xyt_all[:,5] = self.y2 
        xyt_all[:,6] = self.tof2
        xyt_all[:,7] = self.x3 
        xyt_all[:,8] = self.y3
        xyt_all[:,9] = self.tof3
        xyt_all[:,10] = adc1
        xyt_all[:,11] = adc2
        xyt_all[:,12] = index
        np.save(file, xyt_all) #save array as a binary file
        