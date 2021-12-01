# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:13:00 2020
Nathan Marshall

"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from random import gauss
from matplotlib import patches as mpatches
from ColtrimsAnalysis import hist1d, hist2d

class ThreeBodyCE:
    
    def __init__(self, m1, m2, m3, q1, q2, q3, r10, r20, r30, t0, tmax):
        
        r10 = np.asarray(r10)
        r20 = np.asarray(r20)
        r30 = np.asarray(r30)
        
        k = 8.9875517923e9 #Coulomb force constant
        m1 = m1 * 1.66053903e-27 #convert mass from g/mole to kg
        m2 = m2 * 1.66053903e-27
        m3 = m3 * 1.66053903e-27 
        q1 = q1 * 1.60217662e-19
        q2 = q2 * 1.60217662e-19
        q3 = q3 * 1.60217662e-19

        def coul(r1, q1, m1, r2, q2):
            '''Accleration from Coulomb force on charge q1 at r1 by charge q2 at r2.'''
            return k * q1 *q2 * (r1-r2) / np.linalg.norm(r1-r2)**3 / m1
        
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
            dvx1, dvy1, dvz1 = coul(r1, q1, m1, r2, q2) + coul(r1, q1, m1, r3, q3)
            dvx2, dvy2, dvz2 = coul(r2, q2, m2, r1, q1) + coul(r2, q2, m2, r3, q3)
            dvx3, dvy3, dvz3 = coul(r3, q3, m3, r1, q1) + coul(r3, q3, m3, r2, q2)
            
            return(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, 
                   dvx1, dvy1, dvz1, dvx2, dvy2, dvz2, dvx3, dvy3, dvz3)
        
        x10, y10, z10 = r10 #unpack fragment initial position vectors
        x20, y20, z20 = r20
        x30, y30, z30 = r30
        
        ivs = [x10, y10, z10, x20, y20, z20, x30, 
               y30, z30, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        sol = solve_ivp(diffeq, [t0, tmax], ivs)
        self.ke1 = 0.5 * m1 * (sol.y[9]**2 + sol.y[10]**2 + sol.y[11]**2) / 1.60217662e-19
        self.ke2 = 0.5 * m2 * (sol.y[12]**2 + sol.y[13]**2 + sol.y[14]**2) / 1.60217662e-19
        self.ke3 = 0.5 * m3 * (sol.y[15]**2 + sol.y[16]**2 + sol.y[17]**2) / 1.60217662e-19
        self.ker_t = self.ke1 + self.ke2 + self.ke3
        self.t = sol.t
        
        self.ke_tot1 = 0.5 * m1 * (sol.y[9][-1]**2 + sol.y[10][-1]**2 + sol.y[11][-1]**2) / 1.60217662e-19
        self.ke_tot2 = 0.5 * m2 * (sol.y[12][-1]**2 + sol.y[13][-1]**2 + sol.y[14][-1]**2) / 1.60217662e-19
        self.ke_tot3 = 0.5 * m3 * (sol.y[15][-1]**2 + sol.y[16][-1]**2 + sol.y[17][-1]**2) / 1.60217662e-19
        self.px1 = m1 * sol.y[9][-1] / 1.99285191410e-24
        self.py1 = m1 * sol.y[10][-1] / 1.99285191410e-24
        self.pz1 = m1 * sol.y[11][-1] / 1.99285191410e-24
        self.px2 = m2* sol.y[12][-1] / 1.99285191410e-24
        self.py2 = m2 * sol.y[13][-1] / 1.99285191410e-24
        self.pz2 = m2 * sol.y[14][-1] / 1.99285191410e-24
        self.px3 = m3 * sol.y[15][-1] / 1.99285191410e-24
        self.py3 = m3 * sol.y[16][-1] / 1.99285191410e-24
        self.pz3 = m3 * sol.y[17][-1] / 1.99285191410e-24
        
        
        self.p_ion1 = np.array([self.px1, self.py1, self.pz1])
        self.p_ion2 = np.array([self.px2, self.py2, self.pz2])
        self.p_ion3 = np.array([self.px3, self.py3, self.pz3])
        self.ptotx = self.px1 + self.px2 + self.px3
        self.ptoty = self.py1 + self.py2 + self.py3
        self.ptotz = self.pz1 + self.pz2 + self.pz3
        self.ker = self.ke_tot1 + self.ke_tot2 + self.ke_tot3

    def energy(self):
        print('KE 1:', self.ke1[-1], 'eV')
        print('KE 2:', self.ke2[-1], 'eV')
        print('KE 3:', self.ke3[-1], 'eV')
        print('KER: ', self.ker_t[-1], 'eV')
    
    def angles(self):
        theta1 = np.arccos(np.dot(self.p_ion1, self.p_ion2) / 
                           np.linalg.norm(self.p_ion1) / 
                           np.linalg.norm(self.p_ion2))
        theta2 = np.arccos(np.dot(self.p_ion1, self.p_ion3) / 
                           np.linalg.norm(self.p_ion1) / 
                           np.linalg.norm(self.p_ion3))
        theta3 = np.arccos(np.dot(self.p_ion2, self.p_ion3) / 
                           np.linalg.norm(self.p_ion2) / 
                           np.linalg.norm(self.p_ion3))
        
        print('Angle 1-2', theta1 * 180 / np.pi, 'degrees')
        print('Angle 1-3', theta2 * 180 / np.pi, 'degrees')
        print('Angle 2-3', theta3 * 180 / np.pi, 'degrees')
        
    def newton(self, norm=False):
        #relative to ion 1
        mag1 = np.linalg.norm(self.p_ion1)
        mag2 = np.linalg.norm(self.p_ion2)
        mag3 = np.linalg.norm(self.p_ion3)
        dot1_2 = np.dot(self.p_ion1, self.p_ion2)
        dot1_3 = np.dot(self.p_ion1, self.p_ion3)
        
        if norm:
            px2 = dot1_2 / mag1**2
            mag2 = mag2 / mag1
            py2 = (mag2**2 - px2**2)**(1/2)
            px3 = dot1_3 / mag1**2
            mag3 = mag3 / mag1
            py3 = -(mag3**2 - px3**2)**(1/2)
            mag1 = 1
        
        else:
            dot1_2 = np.dot(self.p_ion1, self.p_ion2)
            dot1_3 = np.dot(self.p_ion1, self.p_ion3)
            px2 = dot1_2 / mag1 
            py2 = (mag2**2 - px2**2)**(1/2)
            px3 = dot1_3 / mag1 
            py3 = -(mag3**2 - px3**2)**(1/2)
        
        print('Newton plot values relative to ion 1:')
        print([mag1, 0])
        print([px2, py2])
        print([px3, py3])
        
        fig, ax = plt.subplots()
        ax.scatter([px2, px3, mag1], [py2, py3, 0], marker='x', color='r')
        ax.quiver(px2, py2, angles='xy', scale_units='xy', scale=1, width=0.005)
        ax.quiver(px3, py3, angles='xy', scale_units='xy', scale=1, width=0.005)
        ax.quiver(mag1, 0, angles='xy', scale_units='xy', scale=1, width=0.005)
        ax.set_title('Newton Plot Relative to Ion 1')
        
        if norm:
            ax.set_xlabel('Relative X momentum')
            ax.set_ylabel('Relative Y momentum')
            
        else:
            ax.set_xlabel('X momentum (a.u.)')
            ax.set_ylabel('Y momentum (a.u.)')
            
        max_dim = max(px2, py2, px3, py3, mag1) * 1.25
        ax.set_xlim(-max_dim, max_dim)
        ax.set_ylim(-max_dim, max_dim)
        ax.set_aspect('equal')
        ax.grid()
        
    def dalitz1(self):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_val = (epsilon2 - epsilon1) / (3**(1/2))
        y_val = epsilon3 - 1/3
        print('Dalitz plot data, [(e2 - e1)/sqrt(3), e3 - 1/3]:')
        print([x_val, y_val])
        
    def dalitz2(self):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_val = (epsilon1 - epsilon3) / (3**(1/2))
        y_val = epsilon2 - 1/3
        print('Dalitz plot data, [(e1 - e3)/sqrt(3), e2 - 1/3]:')
        print([x_val, y_val])
    
    def dalitz3(self):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_val = (epsilon2 - epsilon3) / (3**(1/2))
        y_val = epsilon1 - 1/3
        print('Dalitz plot data, [(e2 - e3)/sqrt(3), e1 - 1/3]:')
        print([x_val, y_val])


class DissocThreeBodyCE:
    
    def __init__(self, m1, m2, m3, q1, q2, q3, r10, r20, r30, t0, tmax, dissoc_energy, dissoc_axis, delays, tau):
        
        r10 = np.asarray(r10)
        r20 = np.asarray(r20)
        r30 = np.asarray(r30)
        dissoc_axis = np.asarray(dissoc_axis)
        
        k = 8.9875517923e9 #Coulomb force constant
        m1 = m1 * 1.66053903e-27 #convert mass from g/mole to kg
        m2 = m2 * 1.66053903e-27
        m3 = m3 * 1.66053903e-27 
        q1 = q1 * 1.60217662e-19
        q2 = q2 * 1.60217662e-19
        q3 = q3 * 1.60217662e-19
        
        energy_joules = dissoc_energy * 1.60217662e-19
        self.energies = []
        v1 = (2 * energy_joules / (m1 * (m1 / (m2 + m3) + 1)))**(1/2)
        v2 = (2 * energy_joules / ((m2 + m3) * ((m2 + m3) / m1 + 1)))**(1/2)
            
        def coul(r1, q1, m1, r2, q2):
            '''Accleration from Coulomb force on charge q1 at r1 by charge q2 at r2.'''
            return k * q1 * q2 * (r1-r2) / np.linalg.norm(r1-r2)**3 / m1
        
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
            dvx1, dvy1, dvz1 = coul(r1, q1, m1, r2, q2) + coul(r1, q1, m1, r3, q3)
            dvx2, dvy2, dvz2 = coul(r2, q2, m2, r1, q1) + coul(r2, q2, m2, r3, q3)
            dvx3, dvy3, dvz3 = coul(r3, q3, m3, r1, q1) + coul(r3, q3, m3, r2, q2)
            
            return(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, 
                   dvx1, dvy1, dvz1, dvx2, dvy2, dvz2, dvx3, dvy3, dvz3)
        
        self.size = len(delays)
        self.ke_tot1 = np.zeros(self.size)
        self.ke_tot2 = np.zeros(self.size)
        self.ke_tot3 = np.zeros(self.size)
        self.p_ion1 = np.zeros((self.size, 3))
        self.p_ion2 = np.zeros((self.size, 3))
        self.p_ion3 = np.zeros((self.size, 3))
        self.ker = np.zeros(self.size)
        
        for i in range(len(delays)):
            delta_r = (v1 + v2) * (delays[i] + (np.exp(-delays[i] / tau) - 1) * tau)
            r10_dissoc = r10 + delta_r * dissoc_axis
            
            x10, y10, z10 = r10_dissoc #unpack fragment initial position vectors
            x20, y20, z20 = r20
            x30, y30, z30 = r30
            
            v10 = v1 * (1 - np.exp(-delays[i] / tau)) * dissoc_axis
            v20 = -v2 * (1 - np.exp(-delays[i] / tau)) * dissoc_axis
            v30 = -v2 * (1 - np.exp(-delays[i] / tau)) * dissoc_axis
            
            vx10, vy10, vz10 = v10
            vx20, vy20, vz20 = v20
            vx30, vy30, vz30 = v30
        
            ivs = [x10, y10, z10, x20, y20, z20, x30, y30, z30, vx10,
                   vy10, vz10, vx20, vy20, vz20, vx30, vy30, vz30]
            
            sol = solve_ivp(diffeq, [t0, tmax], ivs)
            
            
            self.ke_tot1[i] = 0.5 * m1 * (sol.y[9][-1]**2 + sol.y[10][-1]**2 + sol.y[11][-1]**2) / 1.60217662e-19
            self.ke_tot2[i] = 0.5 * m2 * (sol.y[12][-1]**2 + sol.y[13][-1]**2 + sol.y[14][-1]**2) / 1.60217662e-19
            self.ke_tot3[i] = 0.5 * m3 * (sol.y[15][-1]**2 + sol.y[16][-1]**2 + sol.y[17][-1]**2) / 1.60217662e-19
            px1 = m1 * sol.y[9][-1] / 1.99285191410e-24
            py1 = m1 * sol.y[10][-1] / 1.99285191410e-24
            pz1 = m1 * sol.y[11][-1] / 1.99285191410e-24
            px2 = m2 * sol.y[12][-1] / 1.99285191410e-24
            py2 = m2 * sol.y[13][-1] / 1.99285191410e-24
            pz2 = m2 * sol.y[14][-1] / 1.99285191410e-24
            px3 = m3 * sol.y[15][-1] / 1.99285191410e-24
            py3 = m3 * sol.y[16][-1] / 1.99285191410e-24
            pz3 = m3 * sol.y[17][-1] / 1.99285191410e-24
            
            
            self.p_ion1[i] = np.array([px1, py1, pz1])
            self.p_ion2[i] = np.array([px2, py2, pz2])
            self.p_ion3[i] = np.array([px3, py3, pz3])
            self.ker[i] = self.ke_tot1[i] + self.ke_tot2[i] + self.ke_tot3[i]
            
        self.theta1 = np.zeros(self.size)
        self.theta2 = np.zeros(self.size)
        self.theta3 = np.zeros(self.size)
        
        for i in range(self.size):
            self.theta1[i] = (np.arccos(np.dot(self.p_ion1[i], self.p_ion2[i]) / 
                               np.linalg.norm(self.p_ion1[i]) / 
                               np.linalg.norm(self.p_ion2[i]))) * 180 / np.pi
            self.theta2[i] = (np.arccos(np.dot(self.p_ion1[i], self.p_ion3[i]) / 
                               np.linalg.norm(self.p_ion1[i]) / 
                               np.linalg.norm(self.p_ion3[i]))) * 180 / np.pi
            self.theta3[i] = (np.arccos(np.dot(self.p_ion2[i], self.p_ion3[i]) / 
                               np.linalg.norm(self.p_ion2[i]) / 
                               np.linalg.norm(self.p_ion3[i]))) * 180 / np.pi
    
    def newton(self, norm=False):
        #relative to ion 1
        mag1 = np.linalg.norm(self.p_ion1)
        mag2 = np.linalg.norm(self.p_ion2)
        mag3 = np.linalg.norm(self.p_ion3)
        dot1_2 = np.dot(self.p_ion1, self.p_ion2)
        dot1_3 = np.dot(self.p_ion1, self.p_ion3)
        
        if norm:
            px2 = dot1_2 / mag1**2
            mag2 = mag2 / mag1
            py2 = (mag2**2 - px2**2)**(1/2)
            px3 = dot1_3 / mag1**2
            mag3 = mag3 / mag1
            py3 = -(mag3**2 - px3**2)**(1/2)
            mag1 = 1
        
        else:
            dot1_2 = np.dot(self.p_ion1, self.p_ion2)
            dot1_3 = np.dot(self.p_ion1, self.p_ion3)
            px2 = dot1_2 / mag1 
            py2 = (mag2**2 - px2**2)**(1/2)
            px3 = dot1_3 / mag1 
            py3 = -(mag3**2 - px3**2)**(1/2)
        
        print('Newton plot values relative to ion 1:')
        print([mag1, 0])
        print([px2, py2])
        print([px3, py3])
        
        fig, ax = plt.subplots()
        ax.scatter([px2, px3, mag1], [py2, py3, 0], marker='x', color='r')
        ax.quiver(px2, py2, angles='xy', scale_units='xy', scale=1, width=0.005)
        ax.quiver(px3, py3, angles='xy', scale_units='xy', scale=1, width=0.005)
        ax.quiver(mag1, 0, angles='xy', scale_units='xy', scale=1, width=0.005)
        ax.set_title('Newton Plot Relative to Ion 1')
        
        if norm:
            ax.set_xlabel('Relative X momentum')
            ax.set_ylabel('Relative Y momentum')
            
        else:
            ax.set_xlabel('X momentum (a.u.)')
            ax.set_ylabel('Y momentum (a.u.)')
            
        max_dim = max(px2, py2, px3, py3, mag1) * 1.25
        ax.set_xlim(-max_dim, max_dim)
        ax.set_ylim(-max_dim, max_dim)
        ax.set_aspect('equal')
        ax.grid()
        
    def dalitz1(self):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_val = (epsilon2 - epsilon1) / (3**(1/2))
        y_val = epsilon3 - 1/3
        print('Dalitz plot data, [(e2 - e1)/sqrt(3), e3 - 1/3]:')
        print([x_val, y_val])
        
    def dalitz2(self):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_val = (epsilon1 - epsilon3) / (3**(1/2))
        y_val = epsilon2 - 1/3
        print('Dalitz plot data, [(e1 - e3)/sqrt(3), e2 - 1/3]:')
        print([x_val, y_val])
    
    def dalitz3(self):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_val = (epsilon2 - epsilon3) / (3**(1/2))
        y_val = epsilon1 - 1/3
        print('Dalitz plot data, [(e2 - e3)/sqrt(3), e1 - 1/3]:')
        print([x_val, y_val])
        

class DissocRotationThreeBodyCE:
    
    def __init__(self, m1, m2, m3, q1, q2, q3, r10, r20, r30, t0, tmax, dissoc_energy, dissoc_axis, tau_v, rotation_axis, rotation_velocity, tau_r, delays):
        
        r10 = np.asarray(r10)
        r20 = np.asarray(r20)
        r30 = np.asarray(r30)
        dissoc_axis = np.asarray(dissoc_axis)
        rotation_axis = np.asarray(rotation_axis)
        
        k = 8.9875517923e9 #Coulomb force constant
        m1 = m1 * 1.66053903e-27 #convert mass from g/mole to kg
        m2 = m2 * 1.66053903e-27
        m3 = m3 * 1.66053903e-27 
        q1 = q1 * 1.60217662e-19
        q2 = q2 * 1.60217662e-19
        q3 = q3 * 1.60217662e-19
        
        energy_joules = dissoc_energy * 1.60217662e-19
        self.energies = []
        v1 = (2 * energy_joules / (m1 * (m1 / (m2 + m3) + 1)))**(1/2)
        v2 = (2 * energy_joules / ((m2 + m3) * ((m2 + m3) / m1 + 1)))**(1/2)
        
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

        def coul(r1, q1, m1, r2, q2):
            '''Accleration from Coulomb force on charge q1 at r1 by charge q2 at r2.'''
            return k * q1 * q2 * (r1-r2) / np.linalg.norm(r1-r2)**3 / m1
        
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
            dvx1, dvy1, dvz1 = coul(r1, q1, m1, r2, q2) + coul(r1, q1, m1, r3, q3)
            dvx2, dvy2, dvz2 = coul(r2, q2, m2, r1, q1) + coul(r2, q2, m2, r3, q3)
            dvx3, dvy3, dvz3 = coul(r3, q3, m3, r1, q1) + coul(r3, q3, m3, r2, q2)
            
            return(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, 
                   dvx1, dvy1, dvz1, dvx2, dvy2, dvz2, dvx3, dvy3, dvz3)
        
        self.size = len(delays)
        self.ke_tot1 = np.zeros(self.size)
        self.ke_tot2 = np.zeros(self.size)
        self.ke_tot3 = np.zeros(self.size)
        self.p_ion1 = np.zeros((self.size, 3))
        self.p_ion2 = np.zeros((self.size, 3))
        self.p_ion3 = np.zeros((self.size, 3))
        self.ker = np.zeros(self.size)
        
        for i in range(len(delays)):
            delta_r = (v1 + v2) * (delays[i] + (np.exp(-delays[i] / tau_v) - 1) * tau_v)
            r10_dissoc = r10 + delta_r * dissoc_axis
            theta = rotation_velocity * (1 - np.exp(-delays[i] / tau_r)) * delays[i]
            rot = rotation_matrix(rotation_axis, theta)
            r20_rot = np.dot(rot, r20)
            r30_rot = np.dot(rot, r30)
            
            x10, y10, z10 = r10_dissoc #unpack fragment initial position vectors
            x20, y20, z20 = r20_rot
            x30, y30, z30 = r30_rot
            
            v10 = v1 * (1 - np.exp(-delays[i] / tau_v)) * dissoc_axis
            v20 = -v2 * (1 - np.exp(-delays[i] / tau_v)) * dissoc_axis
            v30 = -v2 * (1 - np.exp(-delays[i] / tau_v)) * dissoc_axis
            
            vx10, vy10, vz10 = v10
            vx20, vy20, vz20 = v20
            vx30, vy30, vz30 = v30
        
            ivs = [x10, y10, z10, x20, y20, z20, x30, y30, z30, vx10,
                   vy10, vz10, vx20, vy20, vz20, vx30, vy30, vz30]
            
            sol = solve_ivp(diffeq, [t0, tmax], ivs)
            
            
            self.ke_tot1[i] = 0.5 * m1 * (sol.y[9][-1]**2 + sol.y[10][-1]**2 + sol.y[11][-1]**2) / 1.60217662e-19
            self.ke_tot2[i] = 0.5 * m2 * (sol.y[12][-1]**2 + sol.y[13][-1]**2 + sol.y[14][-1]**2) / 1.60217662e-19
            self.ke_tot3[i] = 0.5 * m3 * (sol.y[15][-1]**2 + sol.y[16][-1]**2 + sol.y[17][-1]**2) / 1.60217662e-19
            px1 = m1 * sol.y[9][-1] / 1.99285191410e-24
            py1 = m1 * sol.y[10][-1] / 1.99285191410e-24
            pz1 = m1 * sol.y[11][-1] / 1.99285191410e-24
            px2 = m2 * sol.y[12][-1] / 1.99285191410e-24
            py2 = m2 * sol.y[13][-1] / 1.99285191410e-24
            pz2 = m2 * sol.y[14][-1] / 1.99285191410e-24
            px3 = m3 * sol.y[15][-1] / 1.99285191410e-24
            py3 = m3 * sol.y[16][-1] / 1.99285191410e-24
            pz3 = m3 * sol.y[17][-1] / 1.99285191410e-24
            
            
            self.p_ion1[i] = np.array([px1, py1, pz1])
            self.p_ion2[i] = np.array([px2, py2, pz2])
            self.p_ion3[i] = np.array([px3, py3, pz3])
            self.ker[i] = self.ke_tot1[i] + self.ke_tot2[i] + self.ke_tot3[i]
            
        self.theta1 = np.zeros(self.size)
        self.theta2 = np.zeros(self.size)
        self.theta3 = np.zeros(self.size)
        
        for i in range(self.size):
            self.theta1[i] = (np.arccos(np.dot(self.p_ion1[i], self.p_ion2[i]) / 
                               np.linalg.norm(self.p_ion1[i]) / 
                               np.linalg.norm(self.p_ion2[i]))) * 180 / np.pi
            self.theta2[i] = (np.arccos(np.dot(self.p_ion1[i], self.p_ion3[i]) / 
                               np.linalg.norm(self.p_ion1[i]) / 
                               np.linalg.norm(self.p_ion3[i]))) * 180 / np.pi
            self.theta3[i] = (np.arccos(np.dot(self.p_ion2[i], self.p_ion3[i]) / 
                               np.linalg.norm(self.p_ion2[i]) / 
                               np.linalg.norm(self.p_ion3[i]))) * 180 / np.pi
    
    def newton(self, norm=False):
        #relative to ion 1
        mag1 = np.linalg.norm(self.p_ion1)
        mag2 = np.linalg.norm(self.p_ion2)
        mag3 = np.linalg.norm(self.p_ion3)
        dot1_2 = np.dot(self.p_ion1, self.p_ion2)
        dot1_3 = np.dot(self.p_ion1, self.p_ion3)
        
        if norm:
            px2 = dot1_2 / mag1**2
            mag2 = mag2 / mag1
            py2 = (mag2**2 - px2**2)**(1/2)
            px3 = dot1_3 / mag1**2
            mag3 = mag3 / mag1
            py3 = -(mag3**2 - px3**2)**(1/2)
            mag1 = 1
        
        else:
            dot1_2 = np.dot(self.p_ion1, self.p_ion2)
            dot1_3 = np.dot(self.p_ion1, self.p_ion3)
            px2 = dot1_2 / mag1 
            py2 = (mag2**2 - px2**2)**(1/2)
            px3 = dot1_3 / mag1 
            py3 = -(mag3**2 - px3**2)**(1/2)
        
        print('Newton plot values relative to ion 1:')
        print([mag1, 0])
        print([px2, py2])
        print([px3, py3])
        
        fig, ax = plt.subplots()
        ax.scatter([px2, px3, mag1], [py2, py3, 0], marker='x', color='r')
        ax.quiver(px2, py2, angles='xy', scale_units='xy', scale=1, width=0.005)
        ax.quiver(px3, py3, angles='xy', scale_units='xy', scale=1, width=0.005)
        ax.quiver(mag1, 0, angles='xy', scale_units='xy', scale=1, width=0.005)
        ax.set_title('Newton Plot Relative to Ion 1')
        
        if norm:
            ax.set_xlabel('Relative X momentum')
            ax.set_ylabel('Relative Y momentum')
            
        else:
            ax.set_xlabel('X momentum (a.u.)')
            ax.set_ylabel('Y momentum (a.u.)')
            
        max_dim = max(px2, py2, px3, py3, mag1) * 1.25
        ax.set_xlim(-max_dim, max_dim)
        ax.set_ylim(-max_dim, max_dim)
        ax.set_aspect('equal')
        ax.grid()
        
    def dalitz1(self):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_val = (epsilon2 - epsilon1) / (3**(1/2))
        y_val = epsilon3 - 1/3
        print('Dalitz plot data, [(e2 - e1)/sqrt(3), e3 - 1/3]:')
        print([x_val, y_val])
        
    def dalitz2(self):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_val = (epsilon1 - epsilon3) / (3**(1/2))
        y_val = epsilon2 - 1/3
        print('Dalitz plot data, [(e1 - e3)/sqrt(3), e2 - 1/3]:')
        print([x_val, y_val])
    
    def dalitz3(self):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_val = (epsilon2 - epsilon3) / (3**(1/2))
        y_val = epsilon1 - 1/3
        print('Dalitz plot data, [(e2 - e3)/sqrt(3), e1 - 1/3]:')
        print([x_val, y_val])
        
        
class ThreeBodyCEIncomplete:
    
    def __init__(self, m1, m2, m3, q1, q2, q3, r10, r20, r10_frag, r20_frag, 
                 t0, t_frag, tmax, max_angle, num, ion1, ion2, ion3):
        
        r10 = np.asarray(r10)
        r20 = np.asarray(r20)
        r10_frag = np.asarray(r10_frag)
        r20_frag = np.asarray(r20_frag)
        
        k = 8.9875517923e9 #Coulomb force constant
        m1 = m1 * 1.66053903e-27 #convert mass from g/mole to kg
        m2_init = (m2 + m3) * 1.66053903e-27
        m2_frag = m2 * 1.66053903e-27
        m3_frag = m3 * 1.66053903e-27 
        q1 = q1 * 1.60217662e-19
        q2_init = (q2 + q3) * 1.60217662e-19
        q2_frag = q2 * 1.60217662e-19
        q3_frag = q3 * 1.60217662e-19
        
        self.ion1 = ion1
        self.ion2 = ion2
        self.ion3 = ion3
        self.px1 = np.zeros(num)
        self.py1 = np.zeros(num)
        self.pz1 = np.zeros(num)
        self.px2 = np.zeros(num)
        self.py2 = np.zeros(num)
        self.pz2 = np.zeros(num)
        self.px3 = np.zeros(num)
        self.py3 = np.zeros(num)
        self.pz3 = np.zeros(num)
        self.ke_tot1 = np.zeros(num)
        self.ke_tot2 = np.zeros(num)
        self.ke_tot3 = np.zeros(num)
        
        def rand_vector():
            '''Generates a spherically uniform random unit vector.'''
            vec = np.zeros(3)
            for i in range(3):
                vec[i]= gauss(0, 1)
            vec = vec/np.linalg.norm(vec)
            return(vec)
        
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

        def coul(r1, q1, m1, r2, q2):
            '''Accleration from Coulomb force on charge q1 at r1 by charge q2 at r2.'''
            return k * q1 *q2 * (r1-r2) / np.linalg.norm(r1-r2)**3 / m1
        
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
            dvx1, dvy1, dvz1 = coul(r1, q1, m1, r2, q2_init) 
            dvx2, dvy2, dvz2 = coul(r2, q2_init, m2_init, r1, q1) 
            
            return(vx1, vy1, vz1, vx2, vy2, vz2, dvx1, 
                   dvy1, dvz1, dvx2, dvy2, dvz2)
                   
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
            dvx1, dvy1, dvz1 = coul(r1, q1, m1, r2, q2_frag) + coul(r1, q1, m1, r3, q3_frag)
            dvx2, dvy2, dvz2 = coul(r2, q2_frag, m2_frag, r1, q1) + coul(r2, q2_frag, m2_frag, r3, q3_frag)
            dvx3, dvy3, dvz3 = coul(r3, q3_frag, m3_frag, r1, q1) + coul(r3, q3_frag, m3_frag, r2, q2_frag)
            
            return(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, 
                   dvx1, dvy1, dvz1, dvx2, dvy2, dvz2, dvx3, dvy3, dvz3)
        
        x10, y10, z10 = r10 #unpack fragment initial position vectors
        x20, y20, z20 = r20
        ivs1 = [x10, y10, z10, x20, y20, z20, 0, 0, 0, 0, 0, 0]
        
        for i in range(num):
        
            sol = solve_ivp(diffeq1, [t0, t_frag], ivs1)
            
            r11_out = np.array([sol.y[0][-1], sol.y[1][-1], sol.y[2][-1]])
            r21_out = np.array([sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]])
            v11 = np.array([sol.y[6][-1], sol.y[7][-1], sol.y[8][-1]])
            v21 = np.array([sol.y[9][-1], sol.y[10][-1], sol.y[11][-1]])
            
            axis = rand_vector() #choose random axis
            theta = max_angle * np.random.rand() #choose random angle
            
            r11_rot = np.dot(rotation_matrix(axis, theta), r10_frag)
            r21_rot = np.dot(rotation_matrix(axis, theta), r20_frag)
            
            r11 = r11_out 
            r21 = r21_out + r11_rot
            r31 = r21_out + r21_rot
        
            x11, y11, z11 = r11 #unpack fragment initial position vectors
            x21, y21, z21 = r21
            x31, y31, z31 = r31
            
            vx11, vy11, vz11 = v11
            vx21, vy21, vz21 = v21
            
            #define initial conditions list for the diffeq solver
            ivs2 = [x11, y11, z11, x21, y21, z21, x31, y31, z31, 
                   vx11, vy11, vz11, vx21, vy21, vz21, vx21, vy21, vz21]
            
            sol = solve_ivp(diffeq2, [t_frag, tmax], ivs2)
        
            self.ke_tot1[i] = 0.5 * m1 * (sol.y[9][-1]**2 + sol.y[10][-1]**2 + sol.y[11][-1]**2) / 1.60217662e-19
            self.ke_tot2[i] = 0.5 * m2_frag * (sol.y[12][-1]**2 + sol.y[13][-1]**2 + sol.y[14][-1]**2) / 1.60217662e-19
            self.ke_tot3[i] = 0.5 * m3_frag * (sol.y[15][-1]**2 + sol.y[16][-1]**2 + sol.y[17][-1]**2) / 1.60217662e-19
            self.px1[i] = m1 * sol.y[9][-1] / 1.99285191410e-24
            self.py1[i] = m1 * sol.y[10][-1] / 1.99285191410e-24
            self.pz1[i] = m1 * sol.y[11][-1] / 1.99285191410e-24
            self.px2[i] = m2_frag * sol.y[12][-1] / 1.99285191410e-24
            self.py2[i] = m2_frag * sol.y[13][-1] / 1.99285191410e-24
            self.pz2[i] = m2_frag * sol.y[14][-1] / 1.99285191410e-24
            self.px3[i] = m3_frag * sol.y[15][-1] / 1.99285191410e-24
            self.py3[i] = m3_frag * sol.y[16][-1] / 1.99285191410e-24
            self.pz3[i] = m3_frag * sol.y[17][-1] / 1.99285191410e-24
        
        
        self.p_ion1 = [self.px1, self.py1, self.pz1]
        self.p_ion2 = [self.px2, self.py2, self.pz2]
        self.p_ion3 = [self.px3, self.py3, self.pz3]
        self.ptotx = self.px1 + self.px2 + self.px3
        self.ptoty = self.py1 + self.py2 + self.py3
        self.ptotz = self.pz1 + self.pz2 + self.pz3
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




class ThreeBodyPartialCharges:
    
    def __init__(self, m1, m2, m3, q1, q2, q3, r10, r20, r30, rq10, rq20, rq30, t0, tmax):
        
        r10 = np.asarray(r10)
        r20 = np.asarray(r20)
        r30 = np.asarray(r30)
        rq10 = np.asarray(rq10)
        rq20 = np.asarray(rq20)
        rq30 = np.asarray(rq30)
        
        q1 = np.asarray(q1)
        q2 = np.asarray(q2)
        q3 = np.asarray(q3)
        
        k = 8.9875517923e9 #Coulomb force constant
        m1 = m1 * 1.66053903e-27 #convert mass from g/mole to kg
        m2 = m2 * 1.66053903e-27
        m3 = m3 * 1.66053903e-27 
        q1 = q1 * 1.60217662e-19
        q2 = q2 * 1.60217662e-19
        q3 = q3 * 1.60217662e-19

        def coul(rq1, q1, m1, rq2, q2):
            '''
            Accleration from Coulomb force on partial charges q1 at r1 by 
            partial charges q2 at r2.
            '''
            
            acc = np.zeros(3)
            
            for i in range(len(q1)):
                for j in range(len(q2)):
                    acc += (k * q1[i] *q2[j] * (rq1[i]-rq2[j]) / 
                            np.linalg.norm(rq1[i]-rq2[j])**3 / m1)
            return acc
        
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
            vx1 = d[9]
            vy1 = d[10]
            vz1 = d[11]
            vx2 = d[12]
            vy2 = d[13]
            vz2 = d[14]
            vx3 = d[15]
            vy3 = d[16]
            vz3 = d[17]
            
            r1 = np.array([x1, y1, z1]) #define current position vector
            r2 = np.array([x2, y2, z2])
            r3 = np.array([x3, y3, z3])
            rq1 = rq10 + r1 #define positions of partial charges relative to COM
            rq2 = rq20 + r2
            rq3 = rq30 + r3
            
            #calculate accelerations of each fragment
            dvx1, dvy1, dvz1 = coul(rq1, q1, m1, rq2, q2) + coul(rq1, q1, m1, rq3, q3)
            dvx2, dvy2, dvz2 = coul(rq2, q2, m2, rq1, q1) + coul(rq2, q2, m2, rq3, q3)
            dvx3, dvy3, dvz3 = coul(rq3, q3, m3, rq1, q1) + coul(rq3, q3, m3, rq2, q2)
            
            return(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, 
                   dvx1, dvy1, dvz1, dvx2, dvy2, dvz2, dvx3, dvy3, dvz3)
        
        x10, y10, z10 = r10 #unpack fragment initial position vectors
        x20, y20, z20 = r20
        x30, y30, z30 = r30
        
        ivs = [x10, y10, z10, x20, y20, z20, x30, 
               y30, z30, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        sol = solve_ivp(diffeq, [t0, tmax], ivs)
        self.ke1 = 0.5 * m1 * (sol.y[9]**2 + sol.y[10]**2 + sol.y[11]**2) / 1.60217662e-19
        self.ke2 = 0.5 * m2 * (sol.y[12]**2 + sol.y[13]**2 + sol.y[14]**2) / 1.60217662e-19
        self.ke3 = 0.5 * m3 * (sol.y[15]**2 + sol.y[16]**2 + sol.y[17]**2) / 1.60217662e-19
        self.ker_t = self.ke1 + self.ke2 + self.ke3
        self.t = sol.t
        
        self.ke_tot1 = 0.5 * m1 * (sol.y[9][-1]**2 + sol.y[10][-1]**2 + sol.y[11][-1]**2) / 1.60217662e-19
        self.ke_tot2 = 0.5 * m2 * (sol.y[12][-1]**2 + sol.y[13][-1]**2 + sol.y[14][-1]**2) / 1.60217662e-19
        self.ke_tot3 = 0.5 * m3 * (sol.y[15][-1]**2 + sol.y[16][-1]**2 + sol.y[17][-1]**2) / 1.60217662e-19
        self.px1 = m1 * sol.y[9][-1] / 1.99285191410e-24
        self.py1 = m1 * sol.y[10][-1] / 1.99285191410e-24
        self.pz1 = m1 * sol.y[11][-1] / 1.99285191410e-24
        self.px2 = m2 * sol.y[12][-1] / 1.99285191410e-24
        self.py2 = m2 * sol.y[13][-1] / 1.99285191410e-24
        self.pz2 = m2 * sol.y[14][-1] / 1.99285191410e-24
        self.px3 = m3 * sol.y[15][-1] / 1.99285191410e-24
        self.py3 = m3 * sol.y[16][-1] / 1.99285191410e-24
        self.pz3 = m3 * sol.y[17][-1] / 1.99285191410e-24
        
        
        self.p_ion1 = np.array([self.px1, self.py1, self.pz1])
        self.p_ion2 = np.array([self.px2, self.py2, self.pz2])
        self.p_ion3 = np.array([self.px3, self.py3, self.pz3])
        self.ptotx = self.px1 + self.px2 + self.px3
        self.ptoty = self.py1 + self.py2 + self.py3
        self.ptotz = self.pz1 + self.pz2 + self.pz3
        self.ker = self.ke_tot1 + self.ke_tot2 + self.ke_tot3

    def energy(self):
        print('KE 1:', self.ke1[-1], 'eV')
        print('KE 2:', self.ke2[-1], 'eV')
        print('KE 3:', self.ke3[-1], 'eV')
        print('KER: ', self.ker_t[-1], 'eV')
    
    def angles(self):
        theta1 = np.arccos(np.dot(self.p_ion1, self.p_ion2) / 
                           np.linalg.norm(self.p_ion1) / 
                           np.linalg.norm(self.p_ion2))
        theta2 = np.arccos(np.dot(self.p_ion1, self.p_ion3) / 
                           np.linalg.norm(self.p_ion1) / 
                           np.linalg.norm(self.p_ion3))
        theta3 = np.arccos(np.dot(self.p_ion2, self.p_ion3) / 
                           np.linalg.norm(self.p_ion2) / 
                           np.linalg.norm(self.p_ion3))
        
        print('Angle 1-2', theta1 * 180 / np.pi, 'degrees')
        print('Angle 1-3', theta2 * 180 / np.pi, 'degrees')
        print('Angle 2-3', theta3 * 180 / np.pi, 'degrees')