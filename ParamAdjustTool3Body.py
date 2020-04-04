# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:05:13 2019

Nathan Marshall
Creates a GUI for adjusting COLTRIMS parameters. Supports saving adjusted
parameters to a text file named param_adjust_savestate.txt and reading previous
parameters from this file if it already exists. 
"""
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import tkinter as tk
from ColtrimsAnalysis import hist1d
import numpy as np
from scipy.optimize import curve_fit


class ParameterGUI:
    '''Creates GUI for adjusting COLTRIMS parameters'''
    
    def __init__(self, root, xyt_list, masses, charges, param_list):
        da_to_au = 1822.8885
        l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
        self.param_list = param_list
        self.m1, self.m2, self.m3 = [da_to_au*i for i in masses]
        self.xyt_list = xyt_list
        root.title('Parameter Adjustment Tool')
        root.bind('<Return>', self.refresh)
        plt.style.use('default')
        self.fig = Figure(figsize=(10,5), dpi=100)
        self.ax1 = plt.subplot2grid((1,3),(0,0), fig=self.fig)
        self.ax2 = plt.subplot2grid((1,3),(0,1), fig=self.fig)
        self.ax3 = plt.subplot2grid((1,3),(0,2), fig=self.fig)
        self.ax1.grid(True)
        self.ax2.grid(True)
        self.ax3.grid(True)
        self.t0_entry = tk.Entry(root, width=15)
        self.t0_entry.grid(row=0, column=0)
        self.t0_entry.insert(tk.END, t0)
        self.l_entry = tk.Entry(root, width=15)
        self.l_entry.grid(row=0, column=1)
        self.l_entry.insert(tk.END, l)
        self.z0_entry = tk.Entry(root, width=15)
        self.z0_entry.grid(row=0, column=2)
        self.z0_entry.insert(tk.END, z0)
        self.vz0_entry = tk.Entry(root, width=15)
        self.vz0_entry.grid(row=0, column=3)
        self.vz0_entry.insert(tk.END, vz0)
        self.x_entry = tk.Entry(root, width=15)
        self.x_entry.grid(row=0, column=4)
        self.x_entry.insert(tk.END, x_jet)
        self.vx_entry = tk.Entry(root, width=15)
        self.vx_entry.grid(row=0, column=5)
        self.vx_entry.insert(tk.END, vx_jet)
        self.y_entry = tk.Entry(root, width=15)
        self.y_entry.grid(row=0, column=6)
        self.y_entry.insert(tk.END, y_jet)
        self.vy_entry = tk.Entry(root, width=15)
        self.vy_entry.grid(row=0, column=7)
        self.vy_entry.insert(tk.END, vy_jet)
        self.c_entry = tk.Entry(root, width=15)
        self.c_entry.grid(row=0, column=8)
        self.c_entry.insert(tk.END, C)
        
        self.t0_label = tk.Label(root, text='Change t0')
        self.t0_label.grid(row=1, column=0)
        self.l_label = tk.Label(root, text='Change L')
        self.l_label.grid(row=1, column=1)
        self.z0_label = tk.Label(root, text='Change z0')
        self.z0_label.grid(row=1, column=2)
        self.vz0_label = tk.Label(root, text='Change Vz0')
        self.vz0_label.grid(row=1, column=3)
        self.x_label = tk.Label(root, text='Change Jet X')
        self.x_label.grid(row=1, column=4)
        self.vx_label = tk.Label(root, text='Change Jet Vx')
        self.vx_label.grid(row=1, column=5)
        self.y_label = tk.Label(root, text='Change Jet Y')
        self.y_label.grid(row=1, column=6)
        self.vy_label = tk.Label(root, text='Change Jet Vy')
        self.vy_label.grid(row=1, column=7)
        self.c_label = tk.Label(root, text='Change C')
        self.c_label.grid(row=1, column=8)
        
        self.save_params = tk.Button(root, command=self.write_tofile, 
                                     text='Save Parameters as .txt')
        self.save_params.grid(row=4, column=7)
        self.read_params = tk.Button(root, command=self.read_fromfile, 
                                     text='Load Existing Parameters')
        self.read_params.grid(row=4, column=1)
        self.tog_state = tk.IntVar()
        self.tog_gauss = tk.Checkbutton(root, text='Toggle Gaussian Fit', 
                           variable=self.tog_state)
        self.tog_gauss.grid(row=4, column=8)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)#A tk.DrawingArea
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, columnspan=9)
        self.toolbar_frame = tk.Frame(root)
        self.toolbar_frame.grid(sticky='W', row=4, column=3, columnspan=5)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.plot()
    
    def acc(self, q, l, z0, m, C): #returns the acceleration of the ion
        return (2 * q * (l - z0)) / (m * C**2)
    
    def refresh(self, event): #if enter key is hit, refresh the plot
        self.plot()
        
    def gaussfit(self, x, y, p0, ax):
        '''Performs a simple Gaussian fit and plots it to an axes object.'''
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
        coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)
        xmin, xmax = np.amin(x), np.amax(x)
        hist_x = np.linspace(xmin, xmax, 1000)
        hist_fit = gauss(hist_x, *coeff)
        ax.plot(hist_x, hist_fit)
        # Print the mean and standard deviation:
        s1 = 'Fitted Mean = '+str('%.2f'%coeff[1])
        s2 = 'Fitted SD = '+str('%.2f'%coeff[2])
        ax.text(0.01, 0.95, s=s1, fontsize=7, transform=ax.transAxes)
        ax.text(0.01, 0.92, s=s2, fontsize=7, transform=ax.transAxes)
       
    def plot(self):
        tof1,x1,y1,tof2,x2,y2,tof3,x3,y3,delay,adc1,adc2,index = self.xyt_list
        l = float(self.l_entry.get())
        z0 = float(self.z0_entry.get())
        vz0 = float(self.vz0_entry.get())
        x_jet = float(self.x_entry.get())
        vx_jet =float(self.vx_entry.get())
        y_jet = float(self.y_entry.get())
        vy_jet = float(self.vy_entry.get())
        C = float(self.c_entry.get())
        t0 = float(self.t0_entry.get())
        conv = 0.457102 #conversion factor from mm/ns to atomic units
        self.param_list = [l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0]
        acc1 = self.acc(1, l, z0, self.m1, C) #acceleration of 1st ion
        acc2 = self.acc(1, l, z0, self.m2, C) #acceleration of 2nd ion
        acc3 = self.acc(1, l, z0, self.m3, C) #acceleration of 3rd ion
        vx1_cm = ((x1 - (x_jet))/(tof1 - t0)) - (vx_jet)  
        vy1_cm = ((y1 - (y_jet))/(tof1 - t0)) - (vy_jet) 
        vx2_cm = ((x2 - (x_jet))/(tof2 - t0)) - (vx_jet)
        vy2_cm = ((y2 - (y_jet))/(tof2 - t0)) - (vy_jet)
        vx3_cm = ((x3 - (x_jet))/(tof3 - t0)) - (vx_jet)
        vy3_cm = ((y3 - (y_jet))/(tof3 - t0)) - (vy_jet)
        vz1_cm = (l-z0)/(tof1-t0) - (1/2)*acc1*(tof1-t0) + vz0
        vz2_cm = (l-z0)/(tof2-t0) - (1/2)*acc2*(tof2-t0) + vz0
        vz3_cm = (l-z0)/(tof3-t0) - (1/2)*acc3*(tof3-t0) + vz0
        px1 = self.m1 * vx1_cm * conv
        px2 = self.m2 * vx2_cm * conv
        px3 = self.m3 * vx3_cm * conv
        py1 = self.m1 * vy1_cm * conv
        py2 = self.m2 * vy2_cm * conv
        py3 = self.m3 * vy3_cm * conv
        pz1 = self.m1 * vz1_cm * conv
        pz2 = self.m2 * vz2_cm * conv
        pz3 = self.m3 * vz3_cm * conv
        
        ptotx = px1 + px2 + px3
        ptoty = py1 + py2 + py3
        ptotz = pz1 + pz2 + pz3
       
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        h1, edge1 = hist1d(ptotx, self.ax1, 'X Momentum Sum',
                           'X Momentum (a.u.)','Counts', output=True)
        h2, edge2 = hist1d(ptoty, self.ax2, 'Y Momentum Sum',
                           'Y Momentum (a.u.)','', output=True)
        h3, edge3 = hist1d(ptotz, self.ax3, 'Z Momentum Sum',
                           'Z Momentum (a.u.)', output=True)
        if self.tog_state.get() == 1:
            self.gaussfit(edge1, h1, [1,0,1], self.ax1)
            self.gaussfit(edge2, h2, [1,0,1], self.ax2)
            self.gaussfit(edge3, h3, [1,0,1], self.ax3)
            
        self.canvas.draw()
        
    def write_tofile(self):
        with open('param_adjust_savestate.txt', mode='w') as file:
            for param in self.param_list:
                file.write(str(param))
                file.write('\n')
      
    def read_fromfile(self):
        try:
            with open('param_adjust_savestate.txt', mode='r') as file:
                self.param_list = []
                for line in file:
                    self.param_list.append(float(line))
            l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = self.param_list
            self.t0_entry.delete(0, tk.END)
            self.l_entry.delete(0, tk.END)
            self.z0_entry.delete(0, tk.END)
            self.vz0_entry.delete(0, tk.END)
            self.x_entry.delete(0, tk.END)
            self.vx_entry.delete(0, tk.END)
            self.y_entry.delete(0, tk.END)
            self.vy_entry.delete(0, tk.END)
            self.c_entry.delete(0, tk.END)
            self.t0_entry.insert(tk.END, t0)
            self.l_entry.insert(tk.END, l)
            self.z0_entry.insert(tk.END, z0)
            self.vz0_entry.insert(tk.END, vz0)
            self.x_entry.insert(tk.END, x_jet)
            self.vx_entry.insert(tk.END, vx_jet)
            self.y_entry.insert(tk.END, y_jet)
            self.vy_entry.insert(tk.END, vy_jet)
            self.c_entry.insert(tk.END, C)
        except FileNotFoundError:
            print('FileNotFoundError: No param_adjust_savestate.txt file '
                  'found. Check that such a file exists and is in the correct '
                  'directory.')  
  
