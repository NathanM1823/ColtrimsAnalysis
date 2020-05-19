# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:48:25 2019

Nathan Marshall

This module contains Python functions and classes useful for COLTRIMS data
analysis.
"""
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import colors
from fast_histogram import histogram2d, histogram1d
import numpy as np
from scipy.optimize import curve_fit
import glob
import os

def hist2d(x, y, ax, title='', xlabel='', ylabel='', xbinsize='default',
           ybinsize='default', colorbar=True, grid=False, output=False, 
           color_map='inferno'):
    '''
    Creates a two-dimensional histogram of x and y input arrays. Bin number 
    default is the rounded square root of the length of the x input array.
    To correctly implement the plotting feature of this function, generate
    a matplotlib axes objects to add the plot to.
    Returns: If output is set to True, returns a three-item tuple containing 
    2D histogram, x bin edges, and y bin edges.
    '''
    xmin, xmax = np.amin(x), np.amax(x)
    ymin, ymax = np.amin(y), np.amax(y)
    if xbinsize or ybinsize == 'default':
        xbin_num = int(np.round(np.sqrt(len(x))))
        ybin_num = xbin_num
    if xbinsize and ybinsize != 'default':
        xbin_num = int(np.round((xmax - xmin)/xbinsize))
        ybin_num = int(np.round((ymax - ymin)/ybinsize))
    xedge = np.linspace(xmin, xmax, xbin_num)
    yedge = np.linspace(ymin, ymax, ybin_num)
    H = histogram2d(y, x, [ybin_num, xbin_num], [(ymin, ymax), (xmin, xmax)])
    if ax != None:
        img = ax.pcolorfast(xedge, yedge, H, norm=colors.LogNorm(), 
                            cmap=color_map, picker=1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if colorbar == True:
            cbar = plt.colorbar(img)
            cbar.ax.set_ylabel('Counts')
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', onclick)#on mouse click
        ax.grid(grid)
        plt.show()
    if output == True:
        return (H, xedge, yedge)

def hist1d(arr, ax, title='', xlabel='', ylabel='', binsize='default', 
           log=False, grid=True, output=False, orientation='vertical',
           norm_height=False):
    '''
    Creates a one-dimensional histogram of the input array. Bin number default 
    is the rounded square root of the length of the input array. Optional log
    scale y-axis turned off by default.
    
    Returns: If output is set to True, returns a two-item tuple containing the 
    histogram data and the bin edges for the x-axis.
    '''
    xmin, xmax = np.amin(arr), np.amax(arr)
    if binsize == 'default':
        bin_num = int(np.round(np.sqrt(len(arr))))
    if binsize != 'default':
        bin_num = int(np.round((xmax - xmin)/binsize))
    bin_edges = np.linspace(xmin, xmax, bin_num)
    hist = histogram1d(arr, bin_num, (xmin, xmax))
    if norm_height == True:
        hist = hist/np.amax(hist)
    if ax != None:
        if orientation == 'vertical':
            ax.plot(bin_edges, hist, picker=1)
        else:
            ax.plot(hist, bin_edges, picker=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(grid)
        fig = plt.gcf()
        fig.canvas.mpl_connect('pick_event', onpick) #add point picking
        plt.show()
        if log == True:
            ax.set_yscale('log')
    if output == True:
        return(hist, bin_edges)
        
def onpick(event):
    '''Adds point-picking functionality to line plots, runs on-click.'''
    try: #if line plot and 2D image are on same plot, this exception is needed
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        xpoint, ypoint = xdata[ind], ydata[ind]
        print('Picked X: {}, Picked Y: {}'.format(xpoint, ypoint))
    except AttributeError:
        pass
        
def onclick(event):
    '''Adds MouseEvent functionality to 2D image, runs on-click.'''
    print('Mouse X: {}, Mouse Y: {}'.format(event.xdata, event.ydata))
   
def errorbars(step=1, max_min=False, same_color=True):
    '''Add error bars to a histogram plot. Error is sqrt(number of bins).'''
    fig = plt.gcf()
    ax_list = fig.axes
    for ax in ax_list:
        lines = ax.get_lines()
        for line in lines:
            x = line.get_xdata()
            y = line.get_ydata()
            if max_min == True:
                ymin, ymax = np.amin(y), np.amax(y)
                min_idx = np.where(y==ymin)[0][0]
                max_idx = np.where(y==ymax)[0][0]
                xmin, xmax = x[min_idx], x[max_idx]
                x, y = [xmin, xmax], [ymin, ymax]
            error = np.sqrt(y)
            if same_color == True:
                color = line.get_color()
                ax.errorbar(x, y, yerr=error, marker='.', markevery=step,
                         errorevery=step, capsize=3, ls='none',color=color)
            else:
                ax.errorbar(x, y, yerr=error, marker='.', markevery=step,
                         errorevery=step, capsize=3, ls='none')
        
def masscalc(form):
    '''
    Calculates the mass in daltons of a given molecular formula using masses 
    from an internal dictionary of atomic mass values. \n
    Parameters - \n
    form: a string version of the molecular formula. Examples of input 
    string format - 'C3H6O' (acetone), 'C2H2Br2' (dibromomethane) \n
    Returns - \n
    The mass of the input molecular formula in daltons
    '''
    H = 1.00782503223;
    D = 2.01410177812;
    He =  4.00260325413;
    Li = 7.0160034366;
    C = 12;
    Cthir = 13.003355;
    N = 14.00307400443;
    O = 15.994914619;
    F = 18.99840316273;
    Ne =  19.9924401762;
    S = 31.9720711744;
    Cl = 34.968852682;
    Ar = 39.9623831237;
    Br = 78.9183376;
    I = 126.9044719;
    atom_dict = {'H' : H, 'D' : D, 'He' : He, 'Li' : Li, 'C' : C, 'Cthir' : 
                 Cthir, 'N' : N, 'O' : O, 'F' : F, 'Ne': Ne, 'S' : S, 'Cl' : 
                 Cl, 'Ar' : Ar, 'Br' : Br, 'I' : I}
    cap_bool = [] 
    for i in form: #Create list with True values at index location of uppercase
        cap_bool.append(i.isupper())
    cap_idx = [] #Create a list with index values of True values in cap_bool
    for i in range(len(cap_bool)):
        if cap_bool[i] == True:
             cap_idx.append(i)   
    sliced = [] #Slice string at capital letters and put slices into a list
    for i in range(len(cap_idx)):
        try:
            sliced.append(form[cap_idx[i]:cap_idx[i+1]])
        except IndexError:
            sliced.append(form[cap_idx[i]:])   
    atoms = []
    nums = []   
    for item in sliced: #For each slice, separate alpha/numeric values into
        atom = ''       #atoms and numbers
        num = ''
        for i in item:
            if i.isalpha():
                atom += i 
            elif i.isnumeric():
                num += i
        atoms.append(atom)
        if num == '':
            nums.append('1')
        else:
            nums.append(num)    
    nums = [int(i) for i in nums] #convert string numbers to integers
    mass = 0
    for i in range(len(atoms)): #calculate the mass using atom_dict
        mass += atom_dict[atoms[i]] * nums[i]
    return(mass)

def masscalclist(form_list):
    '''Calculate molecular mass in Da for a list of molecular formulae.'''
    return[masscalc(form) for form in form_list]

def formtex(form_list, charges=None):
    '''
    Converts molecular formula to TeX format. If a list of charges is given as 
    well, these charges will be added to the formula.
    '''
    output = []
    ctr = 0
    for form in form_list:
        rebuilt = ''
        for i in form:
            if i.isnumeric():
                rebuilt += ('_' + i)
            else:
                rebuilt += i
        if charges != None:
            if charges[ctr] == 1:
                rebuilt = '$' + rebuilt + '^+$'
            elif charges[ctr] == 0:
                rebuilt = '$' + rebuilt + '$'
            elif charges[ctr] != 1:
                rebuilt = '$' + rebuilt + '^{' + str(charges[ctr]) + '+}$'
        else:
            rebuilt = '$' + rebuilt + '$' 
        ctr += 1
        output.append(rebuilt)
    if len(output) == 1:
        return output[0]
    else:
        return output

def lineq(point_1, point_2):
    '''
    Returns the slope and y-intercept of the line through two input points.
    '''
    x1, y1 = point_1 #first point
    x2, y2 = point_2 #second point
    m = (y2 - y1)/(x2 - x1) #slope between the points
    c = y1 - m*x1 #intercept of the line
    return m, c
        
def load_param(param_default):
    '''
    Loads COLTRIMS parameters either from a text file or default parameters.
    '''
    param_list = []
    choice = input('Would you like to load COLTRIMS parameters from an '
                   'existing param_adjust_savestate.txt file? (y/n): ')
    if choice == 'y':
        print('Loading parameters from param_adjust_savestate.txt...')
        try:
            with open('param_adjust_savestate.txt', mode='r') as file:
                for line in file:
                    param_list.append(float(line))
                    
        except FileNotFoundError:
            print('FileNotFoundError: No param_adjust_savestate.txt file ' 
                  'found. Check that such a file exists and is in the ' 
                  'correct directory.')
            choice = 'n'
    if choice != 'y':
        print('Loading default parameters instead...')
        param_list = param_default
    return(param_list)

def load_allhits(filename, num_files):
    '''Loads All-Hits COLTRIMS data, returns xyt_list used in this module.'''
    x, y, tof = [], [], []
    delay, adc1, adc2, index = [], [], [], []
    print('File Currently Loading:')
    for file in glob.glob(filename)[0:num_files]:
        print(file)
        s = '<f4'
        dt = np.dtype([('delay', s), ('x', s), ('y', s),
                       ('tof', s), ('adc1', s), ('adc2', s),
                       ('index', s)])
        a = np.fromfile(file, dtype=dt, count=-1)
        x = np.concatenate((x, a['x']))
        y = np.concatenate((y, a['y']))
        tof = np.concatenate((tof, a['tof']))
        delay = np.concatenate((delay, a['delay']))
        adc1 = np.concatenate((adc1, a['adc1']))
        adc2 = np.concatenate((adc2, a['adc2']))
        index = np.concatenate((index, a['index']))
    return[tof, x, y, delay, adc1, adc2, index]
    
def load_2body(filename, num_files):
    '''Loads 2-Body COLTRIMS data and returns xyt_list used in this module.'''
    x1, y1, tof1 = [], [], []
    x2, y2, tof2 = [], [], []
    delay, adc1, adc2, index = [], [], [], []
    print('File Currently Loading:')
    for file in glob.glob(filename)[0:num_files]:
        print(file)
        s = '<f4'
        dt = np.dtype([('delay', s), ('x1', s), ('y1', s),
                       ('tof1', s), ('x2', s), ('y2', s),
                       ('tof2', s), ('adc1', s), ('adc2', s),
                       ('index', s)])
        a = np.fromfile(file, dtype=dt, count=-1)
        x1 = np.concatenate((x1, a['x1']))
        y1 = np.concatenate((y1, a['y1']))
        tof1 = np.concatenate((tof1, a['tof1']))
        x2 = np.concatenate((x2, a['x2']))
        y2 = np.concatenate((y2, a['y2']))
        tof2 = np.concatenate((tof2, a['tof2']))
        delay = np.concatenate((delay, a['delay']))
        adc1 = np.concatenate((adc1, a['adc1']))
        adc2 = np.concatenate((adc2, a['adc2']))
        index = np.concatenate((index, a['index']))
    return[tof1, x1, y1, tof2, x2, y2, delay, adc1, adc2, index]
    
def load_3body(filename, num_files):
    '''Loads 3-Body COLTRIMS data and returns xyt_list used in this module.'''
    x1, y1, tof1 = [], [], []
    x2, y2, tof2 = [], [], []
    x3, y3, tof3 = [], [], []
    delay, adc1, adc2, index = [], [], [], []
    print('File Currently Loading:')
    for file in glob.glob(filename)[0:num_files]:
        print(file)
        s = '<f4'
        dt = np.dtype([('delay', s), ('x1', s), ('y1', s),
                       ('tof1', s), ('x2', s), ('y2', s),
                       ('tof2', s), ('x3', s), ('y3', s),
                       ('tof3', s), ('adc1', s), ('adc2', s),
                       ('index', s)])
        a = np.fromfile(file, dtype=dt, count=-1)
        x1 = np.concatenate((x1, a['x1']))
        y1 = np.concatenate((y1, a['y1']))
        tof1 = np.concatenate((tof1, a['tof1']))
        x2 = np.concatenate((x2, a['x2']))
        y2 = np.concatenate((y2, a['y2']))
        tof2 = np.concatenate((tof2, a['tof2']))
        x3 = np.concatenate((x3, a['x3']))
        y3 = np.concatenate((y3, a['y3']))
        tof3 = np.concatenate((tof3, a['tof3']))
        delay = np.concatenate((delay, a['delay']))
        adc1 = np.concatenate((adc1, a['adc1']))
        adc2 = np.concatenate((adc2, a['adc2']))
        index = np.concatenate((index, a['index']))
    return[tof1, x1, y1, tof2, x2, y2, tof3, x3, y3, delay, adc1, adc2, index]

def load_frag_2body(directory, filename):
    xyt_all = np.load(directory + '/Fragments/' + filename + 
                      '/' + filename + '.npy')
    delay = xyt_all[:,0]
    x1 = xyt_all[:,1] 
    y1 = xyt_all[:,2] 
    tof1 = xyt_all[:,3]
    x2 = xyt_all[:,4] 
    y2 = xyt_all[:,5] 
    tof2 = xyt_all[:,6] 
    adc1 = xyt_all[:,7] 
    adc2 = xyt_all[:,8] 
    index = xyt_all[:,9] 
    return(tof1, x1, y1, tof2, x2, y2, delay, adc1, adc2, index)

def load_frag_3body(directory, filename):
    xyt_all = np.load(directory + '/Fragments/' + filename + 
                      '/' + filename + '.npy')
    delay = xyt_all[:,0]
    delay = xyt_all[:,0]
    x1 = xyt_all[:,1] 
    y1 = xyt_all[:,2] 
    tof1 = xyt_all[:,3]
    x2 = xyt_all[:,4] 
    y2 = xyt_all[:,5] 
    tof2 = xyt_all[:,6] 
    x3 = xyt_all[:,7] 
    y3 = xyt_all[:,8] 
    tof3 = xyt_all[:,9]
    adc1 = xyt_all[:,10] 
    adc2 = xyt_all[:,11] 
    index = xyt_all[:,12] 
    return(tof1, x1, y1, tof2, x2, y2, tof3, x3, y3,
           delay, adc1, adc2, index)

def load_sim_3body(file):
    xyt_all = np.load(file)
    delay = xyt_all[:,0]
    delay = xyt_all[:,0]
    x1 = xyt_all[:,1] 
    y1 = xyt_all[:,2] 
    tof1 = xyt_all[:,3]
    x2 = xyt_all[:,4] 
    y2 = xyt_all[:,5] 
    tof2 = xyt_all[:,6] 
    x3 = xyt_all[:,7] 
    y3 = xyt_all[:,8] 
    tof3 = xyt_all[:,9]
    adc1 = xyt_all[:,10] 
    adc2 = xyt_all[:,11] 
    index = xyt_all[:,12] 
    return(tof1, x1, y1, tof2, x2, y2, tof3, x3, y3,
           delay, adc1, adc2, index)
    
def tof_cal(fragments, charges, tofs, err=False):
    '''Performs TOF calibration, returns values of C and t0.'''
    da_to_au = 1822.8885 #conversion factor from daltons to atomic units
    masses = [da_to_au * m for m in masscalclist(fragments)]
    for i in range(len(masses)):
        masses[i] = masses[i]/charges[i]
    def tof_func(m_q, C, t0):
        return C * np.sqrt(m_q) + t0
    fit = curve_fit(tof_func, xdata=masses, ydata=tofs) #fit to function
    C_opt, t0_opt = fit[0]
    if err == True:
        pcov = fit[1]
        C_err, t0_err = np.sqrt(np.diag(pcov))
        print('C = {} +/- {}'.format(C_opt, C_err))
        print('t0 = {} +/- {}'.format(t0_opt, t0_err))
    return(C_opt, t0_opt)
            
def test_tofcal(C, t0, fragments, charges):
    '''Tests TOF calibration parameters.'''
    ax = plt.gca()
    da_to_au = 1822.8885 #conversion factor from daltons to atomic units
    masses = [da_to_au * m for m in masscalclist(fragments)]
    for i in range(len(masses)):
        masses[i] = masses[i]/charges[i]
    def tof_func(m_q, C, t0):
        return C * np.sqrt(m_q) + t0
    tofs = []
    for mass in masses:
        tofs.append(tof_func(mass, C, t0))
    forms = formtex(fragments, charges)
    ymin, ymax = ax.get_ylim()
    yval = ymax * 0.5
    for i in range(len(tofs)):
        ax.axvline(tofs[i], linestyle='--', linewidth=1.0)
        ax.text(tofs[i], yval, forms[i], horizontalalignment='left')
    
def gaussfit(x, y, p0, ax, disp_sigma=True, return_val=False):
        '''
        Performs a simple Gaussian fit and optionally plots it to an 
        axes object. 
        '''
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
        coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)
        xmin, xmax = np.amin(x), np.amax(x)
        hist_x = np.linspace(xmin, xmax, 1000)
        hist_fit = gauss(hist_x, *coeff)
        mean = coeff[1]
        sigma = coeff[2]
        if ax != None:
            ax.plot(hist_x, hist_fit)
            if disp_sigma == True:
                s1 = 'Fitted Mean = '+str('%.2f'%mean)
                s2 = 'Fitted SD = '+str('%.2f'%sigma)
                ax.text(0.01, 0.95, s=s1, fontsize=7, transform=ax.transAxes)
                ax.text(0.01, 0.92, s=s2, fontsize=7, transform=ax.transAxes)
        if return_val == True:
            return(mean, sigma)
            
def intensity_calc(pz, wavelength):
    '''
    Calculate the intensity of the laser in W/cm^2 given a maximum drift
    momentum in a neon Z momentum histogram. Pz must be given in a.u. and
    wavelength in meters.
    '''
    wavelength = wavelength / 0.52e-10 #0.52e-10 is a conv. factor to a.u.
    omega = 2 * np.pi * 137 / wavelength #137 is the speed of light in a.u.
    Up = pz**2 / 4 
    I = Up * 4 * omega**2 * 3.50e16 #3.50e16 is a conv. factor to W/cm^2
    return I #I is our desired intensity
        
def apply_xytgate(xyt_list, gate):
    '''Used to apply a gate to all XYT variables'''
    for i in range(len(xyt_list)):
        xyt_list[i] = xyt_list[i][gate]
    return(xyt_list)
    
def poly(x, coeff):
    '''Applies a 4th order polynomial fit from np.polyfit to an input array'''
    return(coeff[0]*x**4 + coeff[1]*x**3 + coeff[2]*x**2 
    + coeff[3]*x + coeff[4])
    
def view_gate2body(xyt_list, masses, charges, p_range, offset, param_list,
                      view_range, binsize='default'):
    '''
    Preview of a gate on 2-body coincidence using a polynomial fit generated 
    from a theoretical TOF 1 versus TOF 2 channel. Use this function to fine
    tune the gate, and then use gate_2body with the same gate parameters to 
    actually perform the gate.
    '''
    da_to_au = 1822.8885 #conversion factor from daltons to atomic units
    mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
    tof1, x1, y1, tof2, x2, y2, delay, adc1, adc2, index = xyt_list
    l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
    m1, m2 = [da_to_au*i for i in masses]
    q1, q2 = charges
    pmin, pmax = p_range
    acc1 = (2 * q1 * (l - z0)) / (m1 * C**2) #acceleration of 1st ion
    acc2 = (2 * q2 * (l - z0)) / (m2 * C**2) #acceleration of 2nd ion
    p1 = np.linspace(pmin, pmax, 200)
    p2 = -p1
    v1 = p1/m1/mm_ns_to_au
    v2 = p2/m2/mm_ns_to_au
    t1 = (-(v1-vz0) + np.sqrt((v1-vz0)**2 + 2*acc1*(l - z0)))/acc1 + t0 #TOF 1st ion
    t2 = (-(v2-vz0) + np.sqrt((v2-vz0)**2 + 2*acc2*(l - z0)))/acc2 + t0 #TOF 2nd ion
    coef = np.polyfit(t1, t2, deg=4)
    
    xmin, xmax = view_range[0]
    ymin, ymax = view_range[1]
    condition = ((tof1 > xmin)&(tof1 < xmax)&(tof2 > ymin)&(tof2 < ymax))
    gate = np.where(condition)
    xyt_list = apply_xytgate(xyt_list, gate)
    tof1, x1, y1, tof2, x2, y2, delay, adc1, adc2, index = xyt_list
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title('PIPICO Gate Inspector')
    hist2d(tof1, tof2, ax, 'PIPICO Gate Inspector', 'TOF 1 (ns)', 'TOF 2 (ns)', 
           xbinsize=binsize, ybinsize=binsize)
    ax.plot(t1, poly(t1, coef), 'b')
    ax.plot(t1, poly(t1, coef) + offset, 'w')
    ax.plot(t1, poly(t1, coef) - offset, 'r')
    ax.legend(['Polynomial Fit', 'Upper Gate Bound', 'Lower Gate Bound'])  
    
def view_gated2body(xyt_list, masses, charges, p_range, offset, param_list, 
               binsize='default'):
    '''
    Views 2 body coincidence channel after gating, but does not apply the gate
    to the total xyt array.
    '''
    da_to_au = 1822.8885 #conversion factor from daltons to atomic units
    mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
    tof1, x1, y1, tof2, x2, y2, delay, adc1, adc2, index = xyt_list
    l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
    m1, m2 = [da_to_au*i for i in masses]
    q1, q2 = charges
    pmin, pmax = p_range
    acc1 = (2 * q1 * (l - z0)) / (m1 * C**2) #acceleration of 1st ion
    acc2 = (2 * q2 * (l - z0)) / (m2 * C**2) #acceleration of 2nd ion
    p1 = np.linspace(pmin, pmax, 200)
    p2 = -p1
    v1 = p1/m1/mm_ns_to_au
    v2 = p2/m2/mm_ns_to_au
    t1 = (-(v1-vz0) + np.sqrt((v1-vz0)**2 + 2*acc1*(l - z0)))/acc1 + t0 #TOF 1st ion
    t2 = (-(v2-vz0) + np.sqrt((v2-vz0)**2 + 2*acc2*(l - z0)))/acc2 + t0 #TOF 2nd ion
    coef = np.polyfit(t1, t2, deg=4)
    condition1 = ((tof1 > t1[-1]) & (tof1 < t1[0])
    & (tof2 > t2[0]) & (tof2 < t2[-1])) #Preliminary gate
    gate1 = np.where(condition1) 
    tof1, tof2 = xyt_list[0][gate1], xyt_list[3][gate1]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title('PIPICO Pre-Gate')
    hist2d(tof1, tof2, ax, 'PIPICO Pre-Gate', 'TOF 1 (ns)', 'TOF 2 (ns)', 
           xbinsize=binsize, ybinsize=binsize)
    ax.plot(t1, poly(t1, coef), 'b')
    ax.plot(t1, poly(t1, coef) + offset, 'w')
    ax.plot(t1, poly(t1, coef) - offset, 'r')        
                 
    polytof2 = poly(tof1, coef)
    polyupper = polytof2 + offset
    polylower = polytof2 - offset
    ax.legend(['Polynomial Fit', 'Upper Gate Bound', 'Lower Gate Bound'])
    condition2 = ((tof2 >= polylower) & (tof2 <= polyupper)) #Second gate
    gate2 = np.where(condition2)
    tof1, tof2 = tof1[gate2], tof2[gate2]
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title('PIPICO Post-Gate')
    hist2d(tof1, tof2, ax, 'PIPICO Post-Gate', 'TOF 1 (ns)', 'TOF 2 (ns)', 
           xbinsize=binsize, ybinsize=binsize)
    print(len(gate2[0]), 'Ions Gated')   
    
def gate_2body(xyt_list, masses, charges, p_range, offset, param_list, 
               binsize='default'):
    '''
    Gates on 2-body coincidence using a polynomial fit generated from 
    a theoretical TOF 1 versus TOF 2 channel.
    '''
    da_to_au = 1822.8885 #conversion factor from daltons to atomic units
    mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
    tof1, x1, y1, tof2, x2, y2, delay, adc1, adc2, index = xyt_list
    l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
    m1, m2 = [da_to_au*i for i in masses]
    q1, q2 = charges
    pmin, pmax = p_range
    acc1 = (2 * q1 * (l - z0)) / (m1 * C**2) #acceleration of 1st ion
    acc2 = (2 * q2 * (l - z0)) / (m2 * C**2) #acceleration of 2nd ion
    p1 = np.linspace(pmin, pmax, 200)
    p2 = -p1
    v1 = p1/m1/mm_ns_to_au
    v2 = p2/m2/mm_ns_to_au
    t1 = (-(v1-vz0) + np.sqrt((v1-vz0)**2 + 2*acc1*(l - z0)))/acc1 + t0 #TOF 1st ion
    t2 = (-(v2-vz0) + np.sqrt((v2-vz0)**2 + 2*acc2*(l - z0)))/acc2 + t0 #TOF 2nd ion
    coef = np.polyfit(t1, t2, deg=4)
    condition1 = ((tof1 > t1[-1]) & (tof1 < t1[0])
    & (tof2 > t2[0]) & (tof2 < t2[-1])) #Preliminary gate
    gate1 = np.where(condition1) 
    xyt_list = apply_xytgate(xyt_list, gate1)
    tof1, x1, y1, tof2, x2, y2, delay, adc1, adc2, index = xyt_list   
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title('PIPICO Pre-Gate')
    hist2d(tof1, tof2, ax, 'PIPICO Pre-Gate', 'TOF 1 (ns)', 'TOF 2 (ns)', 
           xbinsize=binsize, ybinsize=binsize)
    ax.plot(t1, poly(t1, coef), 'b')
    ax.plot(t1, poly(t1, coef) + offset, 'w')
    ax.plot(t1, poly(t1, coef) - offset, 'r')        
                 
    polytof2 = poly(tof1, coef)
    polyupper = polytof2 + offset
    polylower = polytof2 - offset
    ax.legend(['Polynomial Fit', 'Upper Gate Bound', 'Lower Gate Bound'])
    condition2 = ((tof2 >= polylower) & (tof2 <= polyupper)) #Second gate
    gate2 = np.where(condition2)
    xyt_list = apply_xytgate(xyt_list, gate2)
    tof1, x1, y1, tof2, x2, y2, delay, adc1, adc2, index = xyt_list
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title('PIPICO Post-Gate')
    hist2d(tof1, tof2, ax, 'PIPICO Post-Gate', 'TOF 1 (ns)', 'TOF 2 (ns)', 
           xbinsize=binsize, ybinsize=binsize)
    print(len(gate2[0]), 'Ions Gated')   
    return xyt_list

def view_gate3body(xyt_list, masses, charges, p_range, offset, param_list,
                      view_range, binsize='default'):
    '''
    Preview of a gate on 3-body coincidence using a polynomial fit generated 
    from a theoretical TOF 1 versus TOF 2 + TOF 3 channel. Use this function 
    to fine-tune the gate, and then use gate_3body with the same gate 
    parameters to actually perform the gate.
    '''
    da_to_au = 1822.8885 #conversion factor from daltons to atomic units
    mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
    tof1, x1, y1, tof2, x2, y2, tof3, x3, y3, delay, adc1, adc2, index = xyt_list
    l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
    m1, m2, m3 = [da_to_au*i for i in masses]
    q1, q2, q3 = charges
    pmin, pmax = p_range
    acc1 = (2 * q1 * (l - z0)) / (m1 * C**2) #acceleration of 1st ion
    acc2 = (2 * q2 * (l - z0)) / (m2 * C**2) #acceleration of 2nd ion
    acc3 = (2 * q3 * (l - z0)) / (m3 * C**2) #acceleration of 2nd ion
    p1 = np.linspace(pmin, pmax, 200)
    p2 = -p1
    p3 = 0
    v1 = p1/m1/mm_ns_to_au
    v2 = p2/m2/mm_ns_to_au
    v3 = p3/m3/mm_ns_to_au
    t1 = (-(v1-vz0) + np.sqrt((v1-vz0)**2 + 2*acc1*(l - z0)))/acc1 + t0 #TOF 1st ion
    t2 = (-(v2-vz0) + np.sqrt((v2-vz0)**2 + 2*acc2*(l - z0)))/acc2 + t0 #TOF 2nd ion
    t3 = (-(v3-vz0) + np.sqrt((v3-vz0)**2 + 2*acc3*(l - z0)))/acc3 + t0 #TOF 3rd ion
    tsum = t2 + t3
    coef = np.polyfit(t1, tsum, deg=4)
    
    xmin, xmax = view_range[0]
    ymin, ymax = view_range[1]
    condition = ((tof1 > xmin)&(tof1 < xmax)&(tof2+tof3 > ymin)&(tof2+tof3 < ymax))
    gate = np.where(condition)
    xyt_list = apply_xytgate(xyt_list, gate)
    tof1, x1, y1, tof2, x2, y2, tof3, x3, y3, delay, adc1, adc2, index = xyt_list
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title('TRIPICO Gate Inspector')
    hist2d(tof1, tof2+tof3, ax, 'TRIPICO Gate Inspector', 'TOF 1 (ns)',
           'TOF 2 + TOF 3 (ns)', xbinsize=binsize, ybinsize=binsize)
    ax.plot(t1, poly(t1, coef), 'b')
    ax.plot(t1, poly(t1, coef) + offset, 'w')
    ax.plot(t1, poly(t1, coef) - offset, 'r')
    ax.legend(['Polynomial Fit', 'Upper Gate Bound', 'Lower Gate Bound']) 
    
def gate_3body(xyt_list, masses, charges, p_range, offset, param_list,
               binsize='default'):
    '''
    Gates on 3-body coincidence using a polynomial fit generated from 
    a theoretical TOF 1 versus TOF 2 + TOF 3 channel.
    '''
    da_to_au = 1822.8885 #conversion factor from daltons to atomic units
    mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
    tof1, x1, y1, tof2, x2, y2, tof3, x3, y3, delay, adc1, adc2, index = xyt_list
    l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
    m1, m2, m3 = [da_to_au*i for i in masses]
    q1, q2, q3 = charges
    pmin, pmax = p_range
    acc1 = (2 * q1 * (l - z0)) / (m1 * C**2) #acceleration of 1st ion
    acc2 = (2 * q2 * (l - z0)) / (m2 * C**2) #acceleration of 2nd ion
    acc3 = (2 * q3 * (l - z0)) / (m3 * C**2) #acceleration of 3rd ion
    p1 = np.linspace(pmin, pmax, 200)
    p2 = -p1
    p3 = 0
    v1 = p1/m1/mm_ns_to_au
    v2 = p2/m2/mm_ns_to_au
    v3 = p3/m3/mm_ns_to_au
    t1 = (-(v1-vz0) + np.sqrt((v1-vz0)**2 + 2*acc1*(l - z0)))/acc1 + t0 #TOF 1st ion
    t2 = (-(v2-vz0) + np.sqrt((v2-vz0)**2 + 2*acc2*(l - z0)))/acc2 + t0 #TOF 2nd ion
    t3 = (-(v3-vz0) + np.sqrt((v3-vz0)**2 + 2*acc3*(l - z0)))/acc3 + t0 #TOF 3rd ion
    tsum = t2 + t3
    coef = np.polyfit(t1, tsum, deg=4)
    condition1 = ((tof1 > t1[-1]) & (tof1 < t1[0]) & (tof2+tof3 > tsum[0]) 
                  & (tof2+tof3 < tsum[-1])) #Preliminary gate
    gate1 = np.where(condition1) 
    xyt_list = apply_xytgate(xyt_list, gate1)
    tof1, x1, y1, tof2, x2, y2, tof3, x3, y3, delay, adc1, adc2, index = xyt_list   
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title('TRIPICO Pre-Gate')
    hist2d(tof1, tof2+tof3, ax, 'TRIPICO Pre-Gate', 'TOF 1 (ns)', 
           'TOF 2 + TOF 3 (ns)', xbinsize=binsize, ybinsize=binsize)
    ax.plot(t1, poly(t1, coef), 'b')
    ax.plot(t1, poly(t1, coef) + offset, 'w')
    ax.plot(t1, poly(t1, coef) - offset, 'r')        
                  
    polytsum = poly(tof1, coef)
    polyupper = polytsum + offset
    polylower = polytsum - offset
    ax.legend(['Polynomial Fit', 'Upper Gate Bound', 'Lower Gate Bound'])
    condition2 = ((tof2+tof3 >= polylower) & (tof2+tof3 <= polyupper))#2nd gate
    gate2 = np.where(condition2)
    xyt_list = apply_xytgate(xyt_list, gate2)
    tof1, x1, y1, tof2, x2, y2, tof3, x3, y3, delay, adc1, adc2, index = xyt_list
    
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title('TRIPICO Post-Gate')
    hist2d(tof1, tof2+tof3, ax, 'TRIPICO Post-Gate', 'TOF 1 (ns)', 
           'TOF 2 + TOF 3 (ns)', xbinsize=binsize, ybinsize=binsize)
    print(len(gate2[0]), 'Ions Gated')   
    return xyt_list

def pipico(xyt_list, tof1range, tof2range, binsize='default'):
    tof1 = xyt_list[0]
    tof2 = xyt_list[3]
    t1min, t1max = tof1range
    t2min, t2max = tof2range
    condition = ((tof1 > t1min) & (tof1 < t1max) & (tof2 > t2min) & 
                 (tof2 < t2max))
    gate = np.where(condition)
    tof1 = tof1[gate]
    tof2 = tof2[gate]
    plt.style.use('dark_background')
    fig = plt.figure('PIPICO Inspection Tool')
    ax1 = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=2, fig=fig)
    hist2d(tof1, tof2, ax1, '', 'TOF 1 (ns)', 'TOF 2 (ns)', 
           xbinsize=binsize, ybinsize=binsize, colorbar=False)
    ax2 = plt.subplot2grid((3,3), (0,0), colspan=2, sharex=ax1, fig=fig)
    hist1d(tof1, ax2, '', '', 'TOF 1 Counts', grid=False, binsize=binsize)
    ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2, sharey=ax1, fig=fig)
    hist1d(tof2, ax3, '', 'TOF 2 Counts', '', orientation='horizontal', 
           grid=False, binsize=binsize)
    ax3.yaxis.tick_right()
    fig.suptitle('PIPICO Inspection Tool', y=0.93, size=14)
    
    def press(event):
        if event.key == ' ':
            gatepipico(tof1, tof2)
    
    def gatepipico(tof1, tof2):
        plt.style.use('dark_background')
        t1min, t1max = ax2.get_xlim()
        t2min, t2max = ax3.get_ylim()
        condition = ((tof1 > t1min) & (tof1 < t1max) & (tof2 > t2min) & 
                     (tof2 < t2max))
        gate = np.where(condition)
        tof1gate = tof1[gate]
        tof2gate = tof2[gate]
        ax2.clear()
        ax3.clear()
        hist1d(tof1gate, ax2, '', '', 'TOF 1 Counts', grid=False, 
               binsize=binsize)
        hist1d(tof2gate, ax3, '', 'TOF 2 Counts', '', 
               orientation='horizontal', grid=False, binsize=binsize)
        fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event', press)
    
def tripico(xyt_list, tof1range, tsumrange, binsize='default'):
    tof1 = xyt_list[0]
    tsum = xyt_list[3] + xyt_list[6]
    t1min, t1max = tof1range
    tsmin, tsmax = tsumrange
    condition = ((tof1 > t1min) & (tof1 < t1max) & (tsum > tsmin) & 
                 (tsum < tsmax))
    gate = np.where(condition)
    tof1 = tof1[gate]
    tsum = tsum[gate]
    plt.style.use('dark_background')
    fig = plt.figure('TRIPICO Inspection Tool')
    ax1 = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=2, fig=fig)
    hist2d(tof1, tsum, ax1, '', 'TOF 1 (ns)', 'TOF 2 + TOF 3 (ns)', 
           xbinsize=binsize, ybinsize=binsize, colorbar=False)
    ax2 = plt.subplot2grid((3,3), (0,0), colspan=2, sharex=ax1, fig=fig)
    hist1d(tof1, ax2, '', '', 'TOF 1 Counts', grid=False, binsize=binsize)
    ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2, sharey=ax1, fig=fig)
    hist1d(tsum, ax3, '', 'TOF 2 + TOF 3 Counts', '', orientation='horizontal', 
           grid=False, binsize=binsize)
    ax3.yaxis.tick_right()
    fig.suptitle('TRIPICO Inspection Tool', y=0.93, size=14)
    
    def press(event):
        if event.key == ' ':
            gatetripico(tof1, tsum)
    
    def gatetripico(tof1, tsum):
        plt.style.use('dark_background')
        t1min, t1max = ax2.get_xlim()
        tsmin, tsmax = ax3.get_ylim()
        condition = ((tof1 > t1min) & (tof1 < t1max) & (tsum > tsmin) & 
                     (tsum < tsmax))
        gate = np.where(condition)
        tof1gate = tof1[gate]
        tsumgate = tsum[gate]
        ax2.clear()
        ax3.clear()
        hist1d(tof1gate, ax2, '', '', 'TOF 1 Counts', grid=False, 
               binsize=binsize)
        hist1d(tsumgate, ax3, '', 'TOF 2 + TOF 3 Counts', '', 
               orientation='horizontal', grid=False, binsize=binsize)
        fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event', press)

class allhits_analysis:
    '''
    This class is used to perform analysis on all hits data from a COLTRIMS
    experiment. \n
    Parameters - \n
    xyt_list: A list containing the all hits analysis data in the format [TOF,
    X, Y, Delay, ADC 1, ADC 2, Index] \n
    molec_name: The name of the molecule being analyzed. This will be used
    in the titles of the plots created.
    '''
    
    def __init__(self, xyt_list, molec_name):
        self.xyt_list = xyt_list
        self.molec_name = molec_name
        
    def gate_tof(self, gate_range, plot=True):
        tof = self.xyt_list[0]
        tofmin, tofmax = gate_range
        condition = ((tof > tofmin) & (tof < tofmax))
        gate = np.where(condition)
        self.xyt_list = apply_xytgate(self.xyt_list, gate)
        self.tof = self.xyt_list[0]
        if plot == True:
            self.tof_hist1d()
    
    def gate_xy(self, xrange, yrange):
        x = self.xyt_list[1]
        y = self.xyt_list[2]
        xmin, xmax = xrange
        ymin, ymax = yrange
        condition = ((x > xmin) & (x < xmax) & (y > ymin) & (y < ymax))
        gate = np.where(condition)
        self.xyt_list = apply_xytgate(self.xyt_list, gate)
        
    def gate_xytof(self, tofrange, xrange, yrange, binsize='default', 
                   return_hist=False, plot_yield=False, ion_form='', 
                   norm=False, gate_all=False, plot_gate=True):
        tof = self.xyt_list[0]
        x = self.xyt_list[1]
        y = self.xyt_list[2]
        tmin, tmax = tofrange
        xmin, xmax = xrange
        ymin, ymax = yrange
        condition = ((tof > tmin) & (tof < tmax) & (x > xmin) & 
                     (x < xmax) & (y > ymin) & (y < ymax))
        gate = np.where(condition)
        tof = tof[gate]
        x = x[gate]
        y = y[gate]
        delay = self.xyt_list[3][gate]
        
        if plot_gate == True:
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.canvas.set_window_title('Ion Gate X/Y/TOF')
            hist2d(tof, x, ax1, 'TOF vs. X {}'.format(ion_form), 'TOF (ns)', 
                   'X (mm)', colorbar=False)
            hist2d(tof, y, ax2, 'TOF vs. Y {}'.format(ion_form), 'TOF (ns)', 
                   'Y (mm)')
            print(gate[0].size,'Ions Gated {}'.format(ion_form))
            
        if plot_yield == True:
            plt.style.use('default')
            fig, ax = plt.subplots(1, 1)
            fig.canvas.set_window_title('Yield vs. Delay {}'.format(
                                         ion_form))
            h, edge = hist1d(delay, ax, 'Yield vs. Delay {}'.format(
                             ion_form),'Delay Index', 'Yield (counts)', 
                             output=True, norm_height=norm, binsize=binsize)
        if gate_all == True:
            self.xyt_list = apply_xytgate(self.xyt_list, gate)
            
        if plot_yield != True:
            h, edge = hist1d(delay, None, output=True, norm_height=norm, 
                             binsize=binsize)
        if return_hist == True:
            return(h, edge)
        
    def tof_hist1d(self, binsize='default', log_scale=True):
        tof = self.xyt_list[0]
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('1D TOF')
        hist1d(tof, ax, 'All Hits TOF Histogram {}'.format(self.molec_name),
               'Time of Flight (ns)', 'Counts', log=log_scale, binsize=binsize)
        
    def detector_img(self, binsize='default'):
        x = self.xyt_list[1]
        y = self.xyt_list[2]
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Detector Image')
        hist2d(x, y, ax, 'All Hits Detector Image - {}'.format(self.molec_name)
               , 'X Position (mm)', 'Y Position (mm)', xbinsize=binsize, 
               ybinsize=binsize)
        
    def tof_x_hist2d(self, tofbin='default', xbin='default'):
        tof = self.xyt_list[0]
        x = self.xyt_list[1]
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('X vs. TOF')
        hist2d(tof, x, ax, 'All Hits X vs. TOF - {}'.format(self.molec_name), 
               'TOF (ns)', 'X Position (mm)', xbinsize=tofbin, ybinsize=xbin)
        
    def tof_y_hist2d(self, tofbin='default', ybin='default'):
        tof = self.xyt_list[0]
        y = self.xyt_list[2]
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Y vs. TOF')
        hist2d(tof, y, ax, 'All Hits Y vs. TOF - {}'.format(self.molec_name), 
               'TOF (ns)', 'Y Position (mm)', xbinsize=tofbin, ybinsize=ybin)
        
    def tof_delay_hist2d(self, delbin='default', tofbin='default'):
        tof = self.xyt_list[0]
        delay = self.xyt_list[3]
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('TOF vs. Delay')
        hist2d(delay, tof, ax, 'All Hits TOF vs. Delay - {}'.format(
               self.molec_name), 'Delay Index', 'TOF (ns)', xbinsize=delbin, 
               ybinsize=tofbin)
            
    def p_ke(self, xyt_list, mass, charge, param_list, ion_form):
        da_to_au = 1822.8885 #conversion factor from daltons to atomic units
        mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
        au_to_ev = 27.211386 #conv. factor from a.u. energy to eV
        self.xyt_list = xyt_list
        self.mass = mass
        self.charge = charge
        self.param_list = param_list
        self.ion1 = ion_form
        self.ion_form = ion_form
        tof1, x1, y1, delay, adc1, adc2, index = xyt_list
        l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
        m1 = da_to_au * mass
        q1 = charge
        self.ion1 = ion_form
        acc1 = (2 * q1 * (l - z0)) / (m1 * C**2) #acceleration of ion
        self.vx1 = ((x1 - (x_jet))/(tof1 - t0)) - (vx_jet)  
        self.vy1 = ((y1 - (y_jet))/(tof1 - t0)) - (vy_jet)
        self.vz1 = (l-z0)/(tof1-t0) - (1/2)*acc1*(tof1-t0) + vz0
        self.px1 = m1 * self.vx1 * mm_ns_to_au
        self.py1 = m1 * self.vy1 * mm_ns_to_au
        self.pz1 = m1 * self.vz1 * mm_ns_to_au
        self.kex1 = self.px1**2 / (2 * m1) * au_to_ev
        self.key1 = self.py1**2 / (2 * m1) * au_to_ev
        self.kez1 = self.pz1**2 / (2 * m1) * au_to_ev
        self.ke_tot = self.kex1 + self.key1 + self.kez1
    
    def plot_pxyz(self, xbin):
        plt.style.use('default')
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
        fig.suptitle('{} Momentum'.format(self.ion1))
        fig.canvas.set_window_title('{} XYZ Momentum'.format(self.ion1))
        hist1d(self.px1, ax1, binsize=xbin)
        hist1d(-(self.px1), ax1, 'X Momentum', 'Momentum (a.u.)', 'Counts',
               binsize=xbin)
        ax1.legend(['$P_x$', '$-P_x$'], loc=1)
        hist1d(self.py1, ax2, binsize=xbin)
        hist1d(-(self.py1), ax2, 'Y Momentum', 'Momentum (a.u.)', binsize=xbin)
        ax2.legend(['$P_y$', '$-P_y$'], loc=1)
        hist1d(self.pz1, ax3, binsize=xbin)
        hist1d(-(self.pz1), ax3, 'Z Momentum', 'Momentum (a.u.)', binsize=xbin)
        ax3.legend(['$P_z$', '$-P_z$'], loc=1)
    
    def plot_pz(self, xbin, return_hist=False):
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('{} Z Momentum'.format(self.ion1))
        hist, edges = hist1d(self.pz1, ax,'{} Z Momentum'.format(self.ion1), 
               'Momentum (a.u.)', 'Counts', binsize=xbin, output=True)
        if return_hist == True:
            return(hist, edges)
            
            
class p_ke_2body:
    '''
    This class is used to perform varius momentum and energy analysis
    operations on X, Y, and time of flight (TOF) data from a COLTRIMS 
    experiment. The data fed into this class must be 2-body coincidence data.\n
    Parameters - \n
    xyt_list: list containing COLTRIMS data in the following format [tof1, x1, 
    y1, tof2, x2, y2, delay, adc1, adc2, index] \n
    masses: list containing the masses of the two ions in the format [mass1, 
    mass2]. The masses of the ions must be given in Daltons. \n
    charges: list containing the charges of the two ions in the format 
    [charge1, charge2]. Charges must be given in units of elementary charge 
    (i.e. integers). \n
    param_list: list containing the COLTRIMS parameters in the format [l, z0, 
    vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0]. \n
    ion_form: list containing the chemical formulae of the ions being analyzed,
    in matplotlib TeX format. For example: [r'$CH_2S^+$', r'$C_3H_2O^+$']
    '''
    def __init__(self, xyt_list, masses, charges, param_list, ion_form):
        da_to_au = 1822.8885 #conversion factor from daltons to atomic units
        mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
        au_to_ev = 27.211386 #conv. factor from a.u. energy to eV
        
        self.xyt_list = xyt_list
        self.masses = masses
        self.charges = charges
        self.param_list = param_list
        self.ion1, self.ion2 = ion_form
        self.ion_form = ion_form
        tof1, x1, y1, tof2, x2, y2, delay, adc1, adc2, index = xyt_list
        l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
        m1, m2 = [da_to_au*i for i in masses]
        q1, q2 = charges
        self.ion1, self.ion2 = ion_form
        acc1 = (2 * q1 * (l - z0)) / (m1 * C**2) #acceleration of 1st ion
        acc2 = (2 * q2 * (l - z0)) / (m2 * C**2) #acceleration of 2nd ion
        self.vx1 = ((x1 - (x_jet))/(tof1 - t0)) - (vx_jet)  
        self.vy1 = ((y1 - (y_jet))/(tof1 - t0)) - (vy_jet)
        self.vz1 = (l-z0)/(tof1-t0) - (1/2)*acc1*(tof1-t0) + vz0
        self.vx2 = ((x2 - (x_jet))/(tof2 - t0)) - (vx_jet)
        self.vy2 = ((y2 - (y_jet))/(tof2 - t0)) - (vy_jet)
        self.vz2 = (l-z0)/(tof2-t0) - (1/2)*acc2*(tof2-t0) + vz0
        self.px1 = m1 * self.vx1 * mm_ns_to_au
        self.py1 = m1 * self.vy1 * mm_ns_to_au
        self.pz1 = m1 * self.vz1 * mm_ns_to_au
        self.px2 = m2 * self.vx2 * mm_ns_to_au
        self.py2 = m2 * self.vy2 * mm_ns_to_au
        self.pz2 = m2 * self.vz2 * mm_ns_to_au
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
       
    def pgate(self, gate_dimen, prange):
        ''' 
        Gates COLTRIMS data in momentum. \n
        Parameters - \n
        gate_dimen: The dimension in which to gate momentum. This must be given
        as a string 'X', 'Y', or 'Z'. \n
        prange: The range of momenta to gate.
        '''
        if gate_dimen == 'X':
            ptot = self.ptotx
        if gate_dimen == 'Y':
            ptot = self.ptoty
        if gate_dimen == 'Z':
            ptot = self.ptotz
        pmin, pmax = prange
        condition = ((ptot > pmin) & (ptot < pmax))
        pgate = np.where(condition)
        self.xyt_list = apply_xytgate(self.xyt_list, pgate)
        self.__init__(self.xyt_list, self.masses, self.charges, 
                      self.param_list, self.ion_form)
        print('{} Ions Gated in {} Momentum'.format(len(pgate[0]), gate_dimen))
        
    def pgate_auto(self, gate_dimen, times_sigma):
        ''' 
        Automatically gates COLTRIMS data in momentum with Gaussian fit. \n
        Parameters - \n
        gate_dimen: The dimension in which to gate momentum. This must be given
        as a string 'X', 'Y', or 'Z'. \n
        times_sigma: The gate size is determined using a Gaussian fit. This
        parameter determines the width of the gate as a multiple of the 
        standard deviation.
        '''
        if gate_dimen == 'X':
            ptot = self.ptotx
        if gate_dimen == 'Y':
            ptot = self.ptoty
        if gate_dimen == 'Z':
            ptot = self.ptotz
        hist, xedge = hist1d(ptot, ax=None, output=True)
        mean, sigma = gaussfit(xedge, hist, [1,0,1], ax=None, return_val=True)
        pmax = times_sigma * sigma
        condition = ((ptot > -pmax) & (ptot < pmax))
        pgate = np.where(condition)
        self.xyt_list = apply_xytgate(self.xyt_list, pgate)
        self.__init__(self.xyt_list, self.masses, self.charges, 
                      self.param_list, self.ion_form)
        print('{} Ions Gated in {} Momentum'.format(len(pgate[0]), gate_dimen))
    
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
        
    def save_frags(self, directory, file):
        if not os.path.exists(directory + '/Fragments'):
            os.makedirs(directory + '/Fragments')
        if not os.path.exists(directory + '/Fragments/' + file):
            os.makedirs(directory + '/Fragments/' + file)
        tof1, x1, y1, tof2, x2, y2, delay, adc1, adc2, index = self.xyt_list
        xyt_all = np.zeros((tof1.size, 10))
        xyt_all[:,0] = delay
        xyt_all[:,1] = x1
        xyt_all[:,2] = y1
        xyt_all[:,3] = tof1
        xyt_all[:,4] = x2
        xyt_all[:,5] = y2
        xyt_all[:,6] = tof2
        xyt_all[:,7] = adc1
        xyt_all[:,8] = adc2
        xyt_all[:,9] = index
        np.save(directory + '/Fragments/' + file + '/' + file, xyt_all)
        
class p_ke_3body:
    '''
    This class is used to perform varius momentum and energy analysis
    operations on X, Y, and time of flight (TOF) data from a COLTRIMS 
    experiment. The data fed into this class must be 3-body coincidence data.\n
    Parameters - \n
    xyt_list: list containing COLTRIMS data in the following format [tof1, x1, 
    y1, tof2, x2, y2, tof3, x3, y3, delay, adc1, adc2, index] \n
    masses: list containing the masses of the two ions in the format [mass1, 
    mass2, mass3]. The masses of the ions must be given in Daltons. \n
    charges: list containing the charges of the two ions in the format 
    [charge1, charge2, charge3]. Charges must be given in units of elementary 
    charge (i.e. integers). \n
    param_list: list containing the COLTRIMS parameters in the format [l, z0, 
    vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0]. \n
    ion_form: list containing the chemical formulae of the ions being analyzed,
    in matplotlib TeX format. For example: [r'$CH_3^+$', r'$C_2H_3^+$',
    r'$C_3H_2^+$']
    '''
    def __init__(self, xyt_list, masses, charges, param_list, ion_form):
        da_to_au = 1822.8885 #conversion factor from daltons to atomic units
        mm_ns_to_au = 0.457102 #conversion factor from mm/ns to atomic units
        au_to_ev = 27.211386 #conv. factor from a.u. energy to eV
        
        self.xyt_list = xyt_list
        self.masses = masses
        self.charges = charges
        self.param_list = param_list
        self.ion1, self.ion2, self.ion3 = ion_form
        self.ion_form = ion_form
        tof1, x1, y1, tof2, x2, y2, tof3, x3, y3, delay, adc1, adc2, index = xyt_list
        l, z0, vz0, x_jet, vx_jet, y_jet, vy_jet, C, t0 = param_list
        self.m1, self.m2, self.m3 = [da_to_au*i for i in masses]
        q1, q2, q3 = charges
        acc1 = (2 * q1 * (l - z0)) / (self.m1 * C**2) #acceleration of 1st ion
        acc2 = (2 * q2 * (l - z0)) / (self.m2 * C**2) #acceleration of 2nd ion
        acc3 = (2 * q3 * (l - z0)) / (self.m3 * C**2) #acceleration of 3rd ion
        self.vx1 = ((x1 - (x_jet))/(tof1 - t0)) - (vx_jet)  
        self.vy1 = ((y1 - (y_jet))/(tof1 - t0)) - (vy_jet)
        self.vz1 = (l-z0)/(tof1-t0) - (1/2)*acc1*(tof1-t0) + vz0
        self.vx2 = ((x2 - (x_jet))/(tof2 - t0)) - (vx_jet)
        self.vy2 = ((y2 - (y_jet))/(tof2 - t0)) - (vy_jet)
        self.vz2 = (l-z0)/(tof2-t0) - (1/2)*acc2*(tof2-t0) + vz0
        self.vx3 = ((x3 - (x_jet))/(tof3 - t0)) - (vx_jet)
        self.vy3 = ((y3 - (y_jet))/(tof3 - t0)) - (vy_jet)
        self.vz3 = (l-z0)/(tof3-t0) - (1/2)*acc3*(tof3-t0) + vz0
        self.px1 = self.m1 * self.vx1 * mm_ns_to_au
        self.py1 = self.m1 * self.vy1 * mm_ns_to_au
        self.pz1 = self.m1 * self.vz1 * mm_ns_to_au
        self.px2 = self.m2 * self.vx2 * mm_ns_to_au
        self.py2 = self.m2 * self.vy2 * mm_ns_to_au
        self.pz2 = self.m2 * self.vz2 * mm_ns_to_au
        self.px3 = self.m3 * self.vx3 * mm_ns_to_au
        self.py3 = self.m3 * self.vy3 * mm_ns_to_au
        self.pz3 = self.m3 * self.vz3 * mm_ns_to_au
        self.p_ion1 = [self.px1, self.py1, self.pz1]
        self.p_ion2 = [self.px2, self.py2, self.pz2]
        self.p_ion3 = [self.px3, self.py3, self.pz3]
        self.ptotx = self.px1 + self.px2 + self.px3
        self.ptoty = self.py1 + self.py2 + self.py3
        self.ptotz = self.pz1 + self.pz2 + self.pz3
        self.kex1 = self.px1**2 / (2 * self.m1) * au_to_ev
        self.kex2 = self.px2**2 / (2 * self.m2) * au_to_ev
        self.kex3 = self.px3**2 / (2 * self.m3) * au_to_ev
        self.key1 = self.py1**2 / (2 * self.m1) * au_to_ev
        self.key2 = self.py2**2 / (2 * self.m2) * au_to_ev
        self.key3 = self.py3**2 / (2 * self.m3) * au_to_ev
        self.kez1 = self.pz1**2 / (2 * self.m1) * au_to_ev
        self.kez2 = self.pz2**2 / (2 * self.m2) * au_to_ev
        self.kez3 = self.pz3**2 / (2 * self.m3) * au_to_ev
        self.ke_tot1 = self.kex1 + self.key1 + self.kez1
        self.ke_tot2 = self.kex2 + self.key2 + self.kez2
        self.ke_tot3 = self.kex3 + self.key3 + self.kez3
        self.ker = self.ke_tot1 + self.ke_tot2 + self.ke_tot3
        
    def pgate(self, gate_dimen, prange):
        ''' 
        Gates COLTRIMS data in momentum. \n
        Parameters - \n
        gate_dimen: The dimension in which to gate momentum. This must be given
        as a string 'X', 'Y', or 'Z'. \n
        prange: The range of momenta to gate.
        '''
        if gate_dimen == 'X':
            ptot = self.ptotx
        if gate_dimen == 'Y':
            ptot = self.ptoty
        if gate_dimen == 'Z':
            ptot = self.ptotz
        pmin, pmax = prange
        condition = ((ptot > pmin) & (ptot < pmax))
        pgate = np.where(condition)
        self.xyt_list = apply_xytgate(self.xyt_list, pgate)
        self.__init__(self.xyt_list, self.masses, self.charges, 
                      self.param_list, self.ion_form)
        print('{} Ions Gated in {} Momentum'.format(len(pgate[0]), gate_dimen))
        
    def pgate_auto(self, gate_dimen, times_sigma):
        ''' 
        Automatically gates COLTRIMS data in momentum with Gaussian fit. \n
        Parameters - \n
        gate_dimen: The dimension in which to gate momentum. This must be given
        as a string 'X', 'Y', or 'Z'. \n
        times_sigma: The gate size is determined using a Gaussian fit. This
        parameter determines the width of the gate as a multiple of the 
        standard deviation.
        '''
        if gate_dimen == 'X':
            ptot = self.ptotx
        if gate_dimen == 'Y':
            ptot = self.ptoty
        if gate_dimen == 'Z':
            ptot = self.ptotz
        hist, xedge = hist1d(ptot, ax=None, output=True)
        mean, sigma = gaussfit(xedge, hist, [1,0,1], ax=None, return_val=True)
        pmax = times_sigma * sigma
        condition = ((ptot > (-pmax + mean)) & (ptot < (pmax + mean)))
        pgate = np.where(condition)
        self.xyt_list = apply_xytgate(self.xyt_list, pgate)
        self.__init__(self.xyt_list, self.masses, self.charges, 
                      self.param_list, self.ion_form)
        print('{} Ions Gated in {} Momentum'.format(len(pgate[0]), gate_dimen))
        
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
    
    def dalitz(self, xbin, ybin):
        epsilon1 = self.ke_tot1 / self.ker
        epsilon2 = self.ke_tot2 / self.ker
        epsilon3 = self.ke_tot3 / self.ker
        x_data = (epsilon2 - epsilon1) / (3**(1/2))
        y_data = epsilon3 - 1/3
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Dalitz Plot')
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
        
    def save_frags(self, directory, file):
        if not os.path.exists(directory + '/Fragments'):
            os.makedirs(directory + '/Fragments')
        if not os.path.exists(directory + '/Fragments/' + file):
            os.makedirs(directory + '/Fragments/' + file)
        tof1, x1, y1, tof2, x2, y2, tof3, x3, y3, delay, adc1, adc2, index = self.xyt_list
        xyt_all = np.zeros((tof1.size, 10))
        xyt_all[:,0] = delay
        xyt_all[:,1] = x1
        xyt_all[:,2] = y1
        xyt_all[:,3] = tof1
        xyt_all[:,4] = x2
        xyt_all[:,5] = y2
        xyt_all[:,6] = tof2
        xyt_all[:,7] = x3
        xyt_all[:,8] = y3
        xyt_all[:,9] = tof3
        xyt_all[:,10] = adc1
        xyt_all[:,11] = adc2
        xyt_all[:,12] = index
        np.save(directory + '/Fragments/' + file + '/' + file, xyt_all)
        