from tokenize import String
import numpy as np

fft = np.fft.fft
ifft = np.fft.ifft

from hardware.waveform import Sin, Waveform, Sequence, Idle, IQWaveform
from hardware.waveform610 import Waveform610

from traits.api import Instance, Property, String, Range, Float, Int, Bool, Array, List, Enum, Trait,\
                        Button, on_trait_change, cached_property, Code
from traitsui.api import View, Item, HGroup, VGroup, Tabbed, VSplit, TextEditor#, EnumEditor, RangeEditor, 
from traitsui.menu import Action, Menu, MenuBar
from traitsui.file_dialog import save_file

from enable.api import Component, ComponentEditor
from chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, CMapImagePlot, BarPlot, DataRange1D, LabelAxis, PlotAxis, cbrewer as COLOR_PALETTE
from chaco.tools.api import PanTool
from chaco.tools.simple_zoom import SimpleZoom

import logging
import time

import analysis.fitting as fitting

from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin, GetSetItemsHandler
from tools.color import scheme

from measurements.pulsed_2awg import Pulsed



# utility functions
def find_laser_pulses(sequence):
    n = 0
    prev = []
    for channels, t in sequence:
        if 'laser' in channels and not 'laser' in prev:
            n += 1
        prev = channels
        if ('sequence' in channels) and (n>0):
            break
    return n

def sequence_length(sequence):
    t = 0
    for c, ti in sequence:
        t += ti
    return t

def sequence_union(s1, s2):
    """
    Return the union of two pulse sequences s1 and s2.
    """
    # make sure that s1 is the longer sequence and s2 is merged into it
    if sequence_length(s1) < sequence_length(s2):
        sp = s2
        s2 = s1
        s1 = sp
    s = []
    c1, dt1 = s1.pop(0)
    c2, dt2 = s2.pop(0)
    while True:
        if dt1 < dt2:
            s.append((set(c1) | set(c2), dt1))
            dt2 -= dt1
            try:
                c1, dt1 = s1.pop(0)
            except:
                break
        elif dt2 < dt1:
            s.append((set(c1) | set(c2), dt2))
            dt1 -= dt2
            try:
                c2, dt2 = s2.pop(0)
            except:
                c2 = []
                dt2 = np.inf
        else:
            s.append((set(c1) | set(c2), dt1))
            try:
                c1, dt1 = s1.pop(0)
            except:
                break
            try:
                c2, dt2 = s2.pop(0)
            except:
                c2 = []
                dt2 = np.inf            
    return s

def sequence_remove_zeros(sequence):
    fil = filter(lambda x: x[1] != 0.0, sequence)
    return fil

def spin_state(c, dt, T, t0=0.0, t1= -1.):
    
    """
    Compute the spin state from a 2D array of count data.
    
    Parameters:
    
        c    = count data
        dt   = time step
        t0   = beginning of integration window relative to the edge
        t1   = None or beginning of integration window for normalization relative to edge
        T    = width of integration window
        
    Returns:
    
        y       = 1D array that contains the spin state
        profile = 1D array that contains the pulse profile
        edge    = position of the edge that was found from the pulse profile
        
    If t1<0, no normalization is performed. If t1>=0, each data point is divided by
    the value from the second integration window and multiplied with the mean of
    all normalization windows.
    """

    profile = c.sum(0)
    edge = fitting.find_edge(profile)
    I = int(round(T / float(dt)))
    i0 = edge + int(round(t0 / float(dt)))
    y = np.empty((c.shape[0],))
    for i, slot in enumerate(c):
        y[i] = slot[i0:i0 + I].sum()
    if t1 >= 0:
        i1 = edge + int(round(t1 / float(dt)))    
        y1 = np.empty((c.shape[0],))
        for i, slot in enumerate(c):
            y1[i] = slot[i1:i1 + I].sum()
        y = y / y1 * y1.mean()
    return y, profile, edge

#fitting function
def ExponentialZeroEstimator(x, y): 
    """Exponential Estimator without offset. a*exp(-x/w) + c"""
    c=y[-1]
    a=y[0]-c
    w=x[-1]*0.5
    return a, w, c

def ExponentialZero(x, a, w, c):
    """Exponential centered at zero.
    
        f = a*exp(-x/w) + c
    
    Parameter:
    
    a    = amplitude
    w    = width
    c    = offset in y-direction
    """
    func = a*np.exp(-x/w)+c
    return func


class VQEHandler(GetSetItemsHandler):
    def saveMatrixPlot(self, info):
        filename = save_file(title='Save Matrix Plot')
        if filename is '':
            return
        else:
            if filename.find('.png') == -1:
                filename = filename + '.png'
            info.object.save_matrix_plot(filename)
    
    def saveAll(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_matrix_plot(filename)
            info.object.save(filename)


class VQE_en(Pulsed):
    '''
    +======================================================================================================+
    |                                                                                                      
    |           VARIATIONAL QUANTUM EIGENSOLVER with tomography                                            
    |                                                                                                      
    |           Laser + rf_x(theta) + rf_y(theta) + TOMO             
    |                                                                                                      
    |                                  
    |           theta : 5 kinds. 1 for this point and 4 for 2 gradients                                    
    |           Tomo : 14 kinds of sequences for matrix elements                                          
    |                                                                                                      
    |           |L|: laser & read 
    |
    |
    | This version is without CPMG protected rf gate, meaning only to check the converting 
    | behavior of rf and only measure the matrix element without pi2 mw pulse before any rf pulse
    | 
    | Devices: AWG520 and smiq for MW and AWG610 for RF
    |
    |                                                                                                     
    | Ref: Decoherence-protected quantum gates for a hybrid solid-state spin register                     
    +======================================================================================================+
    '''
    #todo: j = 2, first mw or rf? 

    check_state = Bool(False, desc='only check state, no gradient', label='check state')



    #awg parameters
    main_dir = String('\VQE', desc='.WFM/.PAT folder in AWG 520', label='Dir.')
    main_dir_rf = String('\VQE', desc='.WFM/.PAT folder in AWG 610', label='rf Dir.')
    amp1  = Range(low=0.0, high=1.0, value=1.0, desc='Waveform amplitude factor', label='wfm amp', auto_set=False, enter_set=True)
    amp1_rf  = Range(low=0.0, high=1.0, value=1.0, desc='Waveform_rf amplitude factor', label='wfm rf amp', auto_set=False, enter_set=True)

    
    #variational quamtum eigensolver
    rot_e_y = Range(low=0., high=1., value=0., desc='rotation of electron spin about y axis', label='Rotation y eletron spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)
    rot_e_x = Range(low=0., high=1., value=0., desc='rotation of electron spin about x axis', label='Rotation x eletron spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)
    rot_n_y = Range(low=0., high=1., value=0., desc='rotation of nuclear spin about y axis', label='Rotation y nuclear spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)
    rot_n_x = Range(low=0., high=1., value=0., desc='rotation of nuclear spin about x axis', label='Rotation x nuclear spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)

    freq_rot_e = Range(low=1, high=1e9, value=48.92e6, desc='microwave frequency for rot e', label='frequency rot_e [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    amp_rot_e = Range(low= 0., high=1., value= 1, desc='amp for rot e', label='amp rot_e[500mVpp]', mode='text', auto_set=False, enter_set=True)
    t_2pi_rot_e = Range(low=1., high=100000., value=200., desc='rabi period 2pi pulse length for for rot e', label='2pi t rot_e[ns]', mode='text', auto_set=False, enter_set=True)


    
    #Qubit controlling parameters
    #The 4 energy levels is defined 3(electron spin:up, nuclear spin:up) 4(up, down) 1(down, up) 2(down, down)
    freq_rf_34 = Range(low=1, high=2.6e9, value=2.940e+6, desc='radio frequency of energy gap between 3 and 4', label='frequency rf_34 [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    amp_rf_34 = Range(low= 0., high=1., value= 1, desc='amp of radiowave of energy gap between 3 and 4', label='amp rf_34[500mVpp]', mode='text', auto_set=False, enter_set=True)
    t_2pi_rf_34 = Range(low=1., high=5000000., value=200.e3, desc='rabi period 2pi pulse length for rf_34', label='2pi t rf_34[ns]', mode='text', auto_set=False, enter_set=True)
    
    freq_rf_12 = Range(low=1, high=2.6e9, value=2.940e+6, desc='radio frequency of energy gap between 1 and 2', label='frequency rf_12 [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    amp_rf_12 = Range(low= 0., high=1., value= 1, desc='amp of radiowave of energy gap between 1 and 2', label='amp rf_12[500mVpp]', mode='text', auto_set=False, enter_set=True)
    t_2pi_rf_12 = Range(low=1., high=5000000., value=200.e3, desc='rabi period 2pi pulse length for rf_12', label='2pi t rf_12[ns]', mode='text', auto_set=False, enter_set=True)
    
    freq_mw_13 = Range(low=1, high=1e9, value=50e6, desc='microwave frequency of energy gap between 1 and 3', label='frequency mw_13[Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    amp_mw_13 = Range(low= 0., high=1., value= 0.04, desc='amp of microwave of energy gap between 1 and 3', label='amp mw_13[500mVpp]', mode='text', auto_set=False, enter_set=True)
    t_2pi_mw_13 = Range(low=1., high=100000., value=1000., desc='rabi period 2pi pulse length for mw_13', label='2pi t mw_13[ns]', mode='text', auto_set=False, enter_set=True)

    freq_mw_24 = Range(low=1, high=1e9, value=47.84e6, desc='microwave frequency of energy gap between 2 and 4', label='frequency mw_24[Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    amp_mw_24 = Range(low= 0., high=1., value= 0.04, desc='amp of microwave of energy gap between 2 and 4', label='amp PP[500mVpp]', mode='text', auto_set=False, enter_set=True)
    t_2pi_mw_24 = Range(low=1., high=100000., value=1000., desc='rabi period 2pi pulse length for mw_24', label='2pi t mw_24[ns]', mode='text', auto_set=False, enter_set=True)

    freq_awg = freq_mw_13


    #result data
    pulse = Array(value=np.array((0., 0.)))
    flank = Float(value=0.0)
    spin_state = Array(value=np.array((0., 0.)))
    spin_state_error = Array(value=np.array((0., 0.)))
    integration_width = Range(low=10., high=4000., value=200., desc='time window for pulse analysis [ns]', label='integr. width [ns]', mode='text', auto_set=False, enter_set=True)
    position_signal = Range(low= -100., high=1000., value=0., desc='position of signal window relative to edge [ns]', label='pos. signal [ns]', mode='text', auto_set=False, enter_set=True)
    position_normalize = Range(low=0., high=10000., value=2200., desc='position of normalization window relative to edge [ns]', label='pos. norm. [ns]', mode='text', auto_set=False, enter_set=True)

    #plotting
    show_raw_data = Bool(False, label='show raw data as matrix plot')
    matrix_plot_data = Instance(ArrayPlotData) #raw data of spin state
    matrix_plot = Instance(Plot, editor=ComponentEditor())

    pulse_plot_data = Instance(ArrayPlotData)
    pulse_plot = Instance(Plot)



    def __init__(self, pulse_generator, time_tagger, microwave, awg, awg_rf, **kwargs):
        self.generate_sequence()
        
        super(VQE_en, self).__init__(pulse_generator, time_tagger, microwave, awg, awg_rf, **kwargs)
        
        
        #create different plots
        self._create_matrix_plot()
        self._create_pulse_plot()
        self._create_line_plot()




    def generate_sequence(self):
        laser = self.laser
        wait  = self.wait
        trig_interval = self.trig_interval
        trig_delay = self.trig_delay
        trig_interval_rf = self.trig_interval_rf
        trig_delay_rf = self.trig_delay_rf


        rot_n_y = self.rot_n_y
        rot_n_x = self.rot_n_x

        t_2pi_rot_e = self.t_2pi_rot_e
        t_2pi_rf_34 = self.t_2pi_rf_34
        t_2pi_rf_12 = self.t_2pi_rf_12
        t_2pi_mw_13 = self.t_2pi_mw_13
        t_2pi_mw_24 = self.t_2pi_mw_24
        t_pi_combined = t_2pi_mw_13/2 + t_2pi_mw_24/2

        rot_sequences = []
        t_rot = (rot_n_y + rot_n_x) * t_2pi_rf_34 
        rot_sequences.append(([ ], t_2pi_rot_e/2 + t_rot )) 
        rot_sequences.append(([ ], t_2pi_rot_e/2 + t_rot - t_2pi_rf_34/4 )) #for ny
        rot_sequences.append(([ ], t_2pi_rot_e/2 + t_rot + t_2pi_rf_34/4 ))
        rot_sequences.append(([ ], t_2pi_rot_e/2 + t_rot - t_2pi_rf_34/4 )) #for nx
        rot_sequences.append(([ ], t_2pi_rot_e/2 + t_rot + t_2pi_rf_34/4 ))

        # 4 kinds of sequence for normlize N1 N3 N4 N2
        norm_sequences = []
        norm_sequences.append(([], 0))   # N1    
        norm_sequences.append(([], t_2pi_rot_e/2)) # N3  
        norm_sequences.append(([], t_2pi_rot_e/2 + t_2pi_rf_34/2)) # N4  
        norm_sequences.append(([], t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2)) # N2  

        # polarization rates
        pol_sequences = []
        pol_sequences.append(([], t_2pi_rf_34/2 + t_2pi_rot_e/2)) # first pi_34 then Big Pi, for e
        pol_sequences.append(([], t_2pi_mw_24/2)) # pi_24 for n
        pol_sequences.append(([], t_2pi_rf_34/2 + t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_mw_24/2)) # pi_34, Big Pi, pi_34, pi_24, for n
        pol_sequences.append(([], t_2pi_rf_34/2 + t_2pi_rot_e/2 + t_2pi_rf_34/2)) # pi_34, Big Pi, pi_34, for n

        # 8 kinds of sequence for tomography (include the cnot gate after rot)
        tomo_sequences = []    
        #diagonal terms
        tomo_sequences.append(([], t_2pi_mw_13/2)) 
        tomo_sequences.append(([], t_2pi_mw_24/2))
        tomo_sequences.append(([], 0))
        tomo_sequences.append(([], t_2pi_mw_13/2 + t_2pi_rf_34/2))
        #non diagonal terms

        # test
        test_sequences = []
        test_sequences.append(([], t_2pi_rf_12/2 + t_2pi_mw_24/2)) # a different way to transfer population to N4


        '''
        test_sequences.append(([], t_2pi_rot_e/2 + t_2pi_rf_34/2)) # wait for t_2pi_rf_34/2 to check if the time cause the N3 < N4 or it is just the case
        test_sequences.append(([], t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2)) # wait for t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2 to check if the time cause the N1 > N2 or it is just the case
        test_sequences.append(([], t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2)) # wait for t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2 to check if the time cause the N1 > N2 or it is just the case
        test_sequences.append(([], t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2)) # wait for t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2 to check if the time cause the N1 > N2 or it is just the case
        test_sequences.append(([], t_2pi_mw_13/2 + t_2pi_mw_24/2 + t_2pi_rf_34/2 + t_2pi_mw_24/2 + t_2pi_mw_13/2)) # wait for t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2 to check if the time cause the N1 > N2 or it is just the case
        test_sequences.append(([], t_2pi_rot_e/2 + t_2pi_rf_34 + t_2pi_rot_e/2)) # wait for t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2 to check if the time cause the N1 > N2 or it is just the case
        test_sequences.append(([], t_2pi_rf_12/2)) # wait for t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2 to check if the time cause the N1 > N2 or it is just the case
        '''

        self.test_n = len(test_sequences)

        sequence = [
            (['aom'], laser),
            (['aom'], (576 + 512) * 1e9 /self.sampling_rf + 450 ),#to be the same with other laser
            ([ ], wait )
        ]
        
        if self.check_state:
            i_range = 1
        else:
            i_range = 5

        max_time = t_2pi_rf_34/2 + t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_mw_24/2 #the third of pol_sequences

        for i in range(i_range): #5 for self and 2 gradient*2
            #sequence.append((['ch2'],  trig_interval))
            for j in range(4): #4 fluorescence rates of the different levels
                if j ==0:
                    sequence.append(([], max_time - 100 - norm_sequences[j][1])) #consider (['sequence'], 100)
                else:
                    sequence.append(([], max_time - norm_sequences[j][1]))
                sequence.append((['awg_rf'],  trig_interval_rf))
                sequence.append(([], trig_delay_rf - trig_delay - trig_interval))
                sequence.append((['awg'], trig_interval))
                sequence.append(([], trig_delay )) # since trig_interval_rf is around 523ns and trig_delay is 50ns
                sequence += [norm_sequences[j]]
                sequence.append((['aom', 'laser'], laser))
                sequence.append((['aom'], (576 + 512) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
                sequence.append(([ ], wait))
            for p in range(4): #4 for polarization rates
                sequence.append(([], max_time - pol_sequences[p][1]))
                sequence.append((['awg_rf'],  trig_interval_rf))
                sequence.append(([], trig_delay_rf - trig_delay - trig_interval))
                sequence.append((['awg'], trig_interval))
                sequence.append(([], trig_delay )) # since trig_interval_rf is around 523ns and trig_delay is 50ns
                sequence += [pol_sequences[p]]
                sequence.append((['aom', 'laser'], laser))
                sequence.append((['aom'], (576 + 512) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
                sequence.append(([ ], wait))
            '''
            for k in range(4): #4 to do tomo diag terms
                sequence.append(([], max_time - rot_sequences[i][1] - tomo_sequences[k][1]))
                sequence.append((['awg_rf'],  trig_interval_rf))
                sequence.append(([], trig_delay_rf - trig_delay - trig_interval))
                sequence.append((['awg'], trig_interval))
                sequence.append(([], trig_delay )) # since trig_interval_rf is around 523ns and trig_delay is 50ns
                sequence += [rot_sequences[i]]
                sequence += [tomo_sequences[k]]
                sequence.append((['aom', 'laser'], laser))
                sequence.append((['aom'], (576 + 512) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
                sequence.append(([ ], wait))
            for m in range(self.test_n): # to do some test
                sequence.append((['awg_rf'],  trig_interval_rf))
                sequence.append(([], trig_delay_rf - trig_delay - trig_interval))
                sequence.append((['awg'], trig_interval))
                sequence.append(([], trig_delay )) # since trig_interval_rf is around 523ns and trig_delay is 50ns
                sequence += [test_sequences[m]]
                sequence.append((['aom', 'laser'], laser))
                sequence.append((['aom'], (576 + 512) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
                sequence.append(([ ], wait))
            '''
        sequence.append((['sequence'], 100))
        return sequence


    def VQE_wfm(self,i,j):
        sub_waves = []
        sub_waves_rf = []

        t_2pi_rot_e = self.t_2pi_rot_e * self.sampling / 1e9
        t_2pi_rot_e_rft = self.t_2pi_rot_e * self.sampling_rf / 1e9

        
        t_2pi_rf_34 = self.t_2pi_rf_34 * self.sampling_rf / 1e9
        t_2pi_rf_34_mwt = self.t_2pi_rf_34 * self.sampling / 1e9
        t_2pi_mw_24 = self.t_2pi_mw_24 * self.sampling / 1e9
        t_2pi_mw_24_rft = self.t_2pi_mw_24 * self.sampling_rf / 1e9
        t_2pi_mw_13 = self.t_2pi_mw_13 * self.sampling / 1e9
        t_2pi_mw_13_rft = self.t_2pi_mw_13 * self.sampling_rf / 1e9
        t_2pi_rf_12 = self.t_2pi_rf_12 * self.sampling_rf / 1e9
        t_2pi_rf_12_mwt = self.t_2pi_rf_12 * self.sampling / 1e9

        t_pi_combined = t_2pi_mw_13/2 + t_2pi_mw_24/2
        t_pi_combined_rft = t_pi_combined * self.sampling_rf / 1e9

        freq_rot_e = self.freq_rot_e/self.sampling
        freq_rf_34 = self.freq_rf_34/self.sampling_rf
        freq_rf_12 = self.freq_rf_12/self.sampling_rf
        freq_rf_34_detune = 2.0e6/self.sampling_rf
        freq_mw_13 = self.freq_mw_13/self.sampling
        freq_mw_24 = self.freq_mw_24/self.sampling
        
        zero = Idle(1)
        idle = Waveform('IDLE', [Idle(256)], sampling=self.sampling)
        idle_mw = IQWaveform('IDLE_mw', [Idle(256)], sampling=self.sampling)
        idle_rf = Waveform610('IDLE_rf', [Idle(512)], sampling=self.sampling_rf)

        # pulse objects


        Pi_13 = [Sin(t_2pi_mw_13/2, freq=freq_mw_13, amp=self.amp_mw_13, phase= 0)]
        Pi_24 = [Sin(t_2pi_mw_24/2, freq=freq_mw_24, amp=self.amp_mw_24, phase= 0)]

        Pi_all_x = [Sin(t_2pi_rot_e/2, freq=freq_rot_e, amp=self.amp_rot_e, phase= 0)]
        Pi_all_combined = Pi_13 + Pi_24
        Pi_all_zero = [Idle(t_2pi_rot_e/2)]

        pi_rf = [Sin(t_2pi_rf_34/2, freq=freq_rf_34, amp=self.amp_rf_34, phase= 0)]

        pi2x_rf_p = [Sin(t_2pi_rf_34/4, freq=freq_rf_34, amp=self.amp_rf_34, phase= 0)]
        pi2x_rf_m = [Sin(t_2pi_rf_34/4, freq=freq_rf_34, amp=self.amp_rf_34, phase= np.pi)]
        pi2y_rf_p = [Sin(t_2pi_rf_34/4, freq=freq_rf_34, amp=self.amp_rf_34, phase= np.pi/2.)]
        pi2y_rf_m = [Sin(t_2pi_rf_34/4, freq=freq_rf_34, amp=self.amp_rf_34, phase= np.pi*3/2.)]



        pi_34_rf = [Sin(t_2pi_rf_34/2, freq=freq_rf_34, amp=self.amp_rf_34, phase= 0)]
        pi_34_rf_detune = [Sin(t_2pi_rf_34/2, freq=freq_rf_34_detune, amp=self.amp_rf_34, phase= 0)]
        pi_34_rf_x2 = [Sin(t_2pi_rf_34, freq=freq_rf_34, amp=self.amp_rf_34, phase= 0)]
        pi_12_rf = [Sin(t_2pi_rf_12/2, freq=freq_rf_12, amp=self.amp_rf_12, phase= 0)]



        
        if i == 0:
            rot_ny_duration = self.rot_n_y  * t_2pi_rf_34
            rot_nx_duration = self.rot_n_x  * t_2pi_rf_34
            rot_rf_mwt = (self.rot_n_y + self.rot_n_x) * t_2pi_rf_34_mwt
        if i == 1:
            rot_ny_duration = (self.rot_n_y + 0.25)  * t_2pi_rf_34
            rot_nx_duration = self.rot_n_x  * t_2pi_rf_34
            rot_rf_mwt = ((self.rot_n_y + 0.25) + self.rot_n_x) * t_2pi_rf_34_mwt
        if i == 2:
            rot_ny_duration = (self.rot_n_y - 0.25)  * t_2pi_rf_34
            rot_nx_duration = self.rot_n_x  * t_2pi_rf_34
            rot_rf_mwt = ((self.rot_n_y - 0.25) + self.rot_n_x) * t_2pi_rf_34_mwt
        if i == 3:
            rot_ny_duration = self.rot_n_y  * t_2pi_rf_34
            rot_nx_duration = (self.rot_n_x + 0.25)  * t_2pi_rf_34
            rot_rf_mwt = (self.rot_n_y + (self.rot_n_x + 0.25)) * t_2pi_rf_34_mwt
        if i == 4:
            rot_ny_duration = self.rot_n_y  * t_2pi_rf_34
            rot_nx_duration = (self.rot_n_x - 0.25)  * t_2pi_rf_34
            rot_rf_mwt = (self.rot_n_y + (self.rot_n_x - 0.25)) * t_2pi_rf_34_mwt
        
        rot_ny = Sin(rot_ny_duration, freq=freq_rf_34, amp=self.amp_rf_34, phase= np.pi/2.)
        rot_nx = Sin(rot_nx_duration, freq=freq_rf_34, amp=self.amp_rf_34, phase= 0)
        
        pulse_rot = [zero, rot_ny, zero, rot_nx, zero]

        if j == 0:
            sub = idle_mw

            sub_rf = idle_rf

            return sub, sub_waves, sub_rf, sub_waves_rf

        elif j == 1:
            #mw
            pulse_sequence = [zero] + Pi_all_x + [zero]
            name = 'VQE_%02i_%02i' % (i, j)
            sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
            sub_waves.append(sub['I'])
            sub_waves.append(sub['Q'])

            #rf
            sub_rf = idle_rf

            return sub, sub_waves, sub_rf, sub_waves_rf
    
        elif j == 2:
            #mw
            pulse_sequence = [zero] + Pi_all_x + [zero]
            name = 'VQE_%02i_%02i' % (i, j)
            sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
            sub_waves.append(sub['I'])
            sub_waves.append(sub['Q'])

            #rf
            if int(t_2pi_rot_e_rft/2) < 512:
                carry = Idle(0)
                carry.duration = int(t_2pi_rot_e_rft/2) 

                pulse_sequence_rf = [carry] + pi_34_rf + [zero]
                name_rf = 'VQE_%02i_%02i_rf' % (i, j)
                sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
                sub_waves_rf.append(sub_rf)
            else: # usually this won't happen
                print('t pi too long')
                sub_rf = Sequence('VQE_%02i_%02i_rf' % (i, j) )

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rot_e_rft/2) / 512, int(t_2pi_rot_e_rft/2) % 512
                
                if n > 0:
                    sub_rf.append(idle_rf, repeat=n)

                pulse_sequence_rf = [carry] + pi_34_rf + [zero]
                name_rf = 'VQE_%02i_%02i_1_rf' % (i, j)
                wfm_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
                sub_waves_rf.append(wfm_rf)

                sub_rf.append(wfm_rf)
                sub_waves_rf.append(sub_rf)
                
            return sub, sub_waves, sub_rf, sub_waves_rf
            
        elif j == 3:
            #mw
            pulse_sequence = [zero] + Pi_all_x + [zero]
            wfm1 = IQWaveform('VQE_%02i_%02i_1'% (i, j), pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
            sub_waves.append(wfm1['I'])
            sub_waves.append(wfm1['Q'])

            t_b, stub = wfm1['I'].duration, wfm1['I'].stub

            if int(t_2pi_rf_34_mwt/2) < stub:
                carry = Idle(0)
                carry.duration = int(t_2pi_rf_34_mwt/2)
                pulse_sequence = [zero] + Pi_all_x + [carry] + Pi_all_x + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )
                sub.append(wfm1)

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rf_34_mwt/2 - stub) / 256, int(t_2pi_rf_34_mwt/2 - stub) % 256
                t_offset = t_b + n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence = [carry] + Pi_all_x + [zero]
                wfm2 = IQWaveform('VQE_%02i_%02i_2'% (i, j), pulse_sequence, t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm2['I'])
                sub_waves.append(wfm2['Q'])
                sub.append(wfm2)
                sub_waves.append(sub)



            #rf
            if int(t_2pi_rot_e_rft/2) < 512:
                carry = Idle(0)
                carry.duration = int(t_2pi_rot_e_rft/2) 

                pulse_sequence_rf = [carry] + pi_34_rf + [zero]
                name_rf = 'VQE_%02i_%02i_rf' % (i, j)
                sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
                sub_waves_rf.append(sub_rf)
            else:# usually this won't happen
                print('t_2pi_rot_e_rft/2 too long!')
                sub_rf = Sequence('VQE_%02i_%02i_rf' % (i, j) )

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rot_e_rft/2) / 512, int(t_2pi_rot_e_rft/2) % 512
                
                if n > 0:
                    sub_rf.append(idle_rf, repeat=n)

                pulse_sequence_rf = [carry] + pi_34_rf + [zero]
                name_rf = 'VQE_%02i_%02i_1_rf' % (i, j)
                wfm_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
                sub_waves_rf.append(wfm_rf)

                sub_rf.append(wfm_rf)
                sub_waves_rf.append(sub_rf)
                
            return sub, sub_waves, sub_rf, sub_waves_rf

  
        elif j == 4:
            #mw
            if int(t_2pi_rf_34_mwt/2) < 256:
                carry = Idle(0)
                carry.duration = int(t_2pi_rf_34_mwt/2)
                pulse_sequence = [carry] + Pi_all_x + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rf_34_mwt/2) / 256, int(t_2pi_rf_34_mwt/2 ) % 256
                t_offset = n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence = [carry] + Pi_all_x + [zero]
                wfm1 = IQWaveform('VQE_%02i_%02i_1'% (i, j), pulse_sequence, t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm1['I'])
                sub_waves.append(wfm1['Q'])
                sub.append(wfm1)
                sub_waves.append(sub)



            #rf
            pulse_sequence_rf = [zero] + pi_34_rf + [zero]
            name_rf = 'VQE_%02i_%02i_rf' % (i, j)
            sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
            sub_waves_rf.append(sub_rf)
            
                
            return sub, sub_waves, sub_rf, sub_waves_rf

        elif j == 5:
            #mw
            pulse_sequence = [zero] + Pi_24 + [zero]
            name = 'VQE_%02i_%02i' % (i, j)
            sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
            sub_waves.append(sub['I'])
            sub_waves.append(sub['Q'])

            #rf
            sub_rf = idle_rf

            return sub, sub_waves, sub_rf, sub_waves_rf


  
        elif j == 6:
            #mw
            if int(t_2pi_rf_34_mwt/2) < 256:# usually this won't happen
                carry = Idle(0)
                carry.duration = int(t_2pi_rf_34_mwt/2)
                pulse_sequence = [carry] + Pi_all_x + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rf_34_mwt/2) / 256, int(t_2pi_rf_34_mwt/2 ) % 256
                t_offset = n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence = [carry] + Pi_all_x + [zero]
                wfm1 = IQWaveform('VQE_%02i_%02i_1'% (i, j), pulse_sequence, t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm1['I'])
                sub_waves.append(wfm1['Q'])
                sub.append(wfm1)
                sub_waves.append(sub)



            #rf
            carry_rf = Idle(0)
            carry_rf.duration = int(t_2pi_rot_e_rft/2) 
            pulse_sequence_rf = [zero] + pi_34_rf + [carry_rf] + pi_34_rf + [zero]
            name_rf = 'VQE_%02i_%02i_rf' % (i, j)
            sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
            sub_waves_rf.append(sub_rf)
            
                
            return sub, sub_waves, sub_rf, sub_waves_rf


  
        elif j == 7:
            #mw
            if int(t_2pi_rf_34_mwt/2) < 256:# usually this won't happen
                carry = Idle(0)
                carry.duration = int(t_2pi_rf_34_mwt/2)
                pulse_sequence = [carry] + Pi_all_x + [carry] + Pi_24 + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )

                carry1 = Idle(0)
                n1, carry1.duration = int(t_2pi_rf_34_mwt/2) / 256, int(t_2pi_rf_34_mwt/2 ) % 256
                t_offset1 = n1 * 256

                if n1 > 0:
                    sub.append(idle, idle, repeat=n1)

                pulse_sequence1 = [carry1] + Pi_all_x
                wfm1 = IQWaveform('VQE_%02i_%02i_1'% (i, j), pulse_sequence1, t_offset1, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm1['I'])
                sub_waves.append(wfm1['Q'])
                sub.append(wfm1)

                t_b, stub = wfm1['I'].duration, wfm1['I'].stub


                carry2 = Idle(0)
                n2, carry2.duration = int(t_2pi_rf_34_mwt/2 - stub) / 256, int(t_2pi_rf_34_mwt/2 - stub) % 256
                t_offset2 = t_b + n2 * 256

                if n2 > 0:
                    sub.append(idle, idle, repeat=n2)

                pulse_sequence2 = [carry2] + Pi_24 + [zero]
                wfm2 = IQWaveform('VQE_%02i_%02i_2'% (i, j), pulse_sequence2, t_offset2, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm2['I'])
                sub_waves.append(wfm2['Q'])
                sub.append(wfm2)

                sub_waves.append(sub)



            #rf
            carry_rf = Idle(0)
            carry_rf.duration = int(t_2pi_rot_e_rft/2) 
            pulse_sequence_rf = [zero] + pi_34_rf + [carry_rf] + pi_34_rf + [zero]
            name_rf = 'VQE_%02i_%02i_rf' % (i, j)
            sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
            sub_waves_rf.append(sub_rf)
            
                
            return sub, sub_waves, sub_rf, sub_waves_rf

 
        elif j == 8:
            #mw
            if int(t_2pi_rot_e/2 + rot_rf_mwt) < 256:
                carry = Idle(0)
                carry.duration = int(rot_rf_mwt) 

                pulse_sequence = [zero] + Pi_all_x + [carry] + Pi_13 + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )
                pulse_sequence1 = [zero] + Pi_all_x
                name1 = 'VQE_%02i_%02i_1' % (i, j)
                wfm1 = IQWaveform(name1, pulse_sequence1, offset=0, file_type=0, sampling=self.sampling)
                sub.append(wfm1)
                sub_waves.append(wfm1['I'])
                sub_waves.append(wfm1['Q'])

                t_b, stub = wfm1['I'].duration, wfm1['I'].stub

                carry = Idle(0)
                n, carry.duration = int(rot_rf_mwt - stub) / 256, int(rot_rf_mwt - stub) % 256
                t_offset = t_b + n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence2 = [carry] + Pi_13 + [zero]
                name2 = 'VQE_%02i_%02i_2' % (i, j)
                wfm2 = IQWaveform(name2, pulse_sequence2, offset=t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm2['I'])
                sub_waves.append(wfm2['Q'])
                sub.append(wfm2)
                sub_waves.append(sub)
            
            #rf
            carry_rf = Idle(0)
            carry_rf.duration = int(t_2pi_rot_e_rft/2) 
            pulse_sequence_rf = [carry_rf, rot_ny, rot_nx, zero]
            #pulse_sequence_rf = [carry_rf] + pulse_rot
            name_rf = 'VQE_%02i_%02i_rf' % (i, j)
            sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
            sub_waves_rf.append(sub_rf)

            return sub, sub_waves, sub_rf, sub_waves_rf


        elif j == 9:
            #mw
            if int(t_2pi_rot_e/2 + rot_rf_mwt) < 256:
                carry = Idle(0)
                carry.duration = int(rot_rf_mwt) 

                pulse_sequence = [zero] + Pi_all_x + [carry] + Pi_24  + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )
                pulse_sequence1 = [zero] + Pi_all_x
                name1 = 'VQE_%02i_%02i_1' % (i, j)
                wfm1 = IQWaveform(name1, pulse_sequence1, offset=0, file_type=0, sampling=self.sampling)
                sub.append(wfm1)
                sub_waves.append(wfm1['I'])
                sub_waves.append(wfm1['Q'])

                t_b, stub = wfm1['I'].duration, wfm1['I'].stub

                carry = Idle(0)
                n, carry.duration = int(rot_rf_mwt - stub) / 256, int(rot_rf_mwt - stub) % 256
                t_offset = t_b + n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence2 = [carry] + Pi_24 + [zero]
                name2 = 'VQE_%02i_%02i_2' % (i, j)
                wfm2 = IQWaveform(name2, pulse_sequence2, offset=t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm2['I'])
                sub_waves.append(wfm2['Q'])
                sub.append(wfm2)
                sub_waves.append(sub)
            
            #rf
            carry_rf = Idle(0)
            carry_rf.duration = int(t_2pi_rot_e_rft/2) 
            pulse_sequence_rf = [carry_rf, rot_ny, rot_nx, zero]
            name_rf = 'VQE_%02i_%02i_rf' % (i, j)
            sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
            sub_waves_rf.append(sub_rf)

            return sub, sub_waves, sub_rf, sub_waves_rf

            
        elif j == 10:
            #mw
            pulse_sequence = [zero] + Pi_all_x + [zero]
            name = 'VQE_%02i_%02i' % (i, j)
            sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
            sub_waves.append(sub['I'])
            sub_waves.append(sub['Q'])
            
            
            #rf
            carry_rf = Idle(0)
            carry_rf.duration = int(t_2pi_rot_e_rft/2) 
            pulse_sequence_rf = [carry_rf, rot_ny, rot_nx, zero]
            name_rf = 'VQE_%02i_%02i_rf' % (i, j)
            sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
            sub_waves_rf.append(sub_rf)

            return sub, sub_waves, sub_rf, sub_waves_rf

            
        elif j == 11:
            #mw
            if int(t_2pi_rot_e/2 + rot_rf_mwt) < 256:
                carry = Idle(0)
                carry.duration = int(rot_rf_mwt) 

                pulse_sequence = [zero] + Pi_all_x + [carry] + Pi_13 + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )
                pulse_sequence1 = [zero] + Pi_all_x
                name1 = 'VQE_%02i_%02i_1' % (i, j)
                wfm1 = IQWaveform(name1, pulse_sequence1, offset=0, file_type=0, sampling=self.sampling)
                sub.append(wfm1)
                sub_waves.append(wfm1['I'])
                sub_waves.append(wfm1['Q'])

                t_b, stub = wfm1['I'].duration, wfm1['I'].stub

                carry = Idle(0)
                n, carry.duration = int(rot_rf_mwt - stub) / 256, int(rot_rf_mwt - stub) % 256
                t_offset = t_b + n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence2 = [carry] + Pi_13 + [zero]
                name2 = 'VQE_%02i_%02i_2' % (i, j)
                wfm2 = IQWaveform(name2, pulse_sequence2, offset=t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm2['I'])
                sub_waves.append(wfm2['Q'])
                sub.append(wfm2)
                sub_waves.append(sub)
          
            
            #rf
            carry1_rf = Idle(0)
            carry1_rf.duration = int(t_2pi_rot_e_rft/2) 
            carry2_rf = Idle(0)
            carry2_rf.duration = int(t_2pi_mw_13_rft/2) 
            pulse_sequence_rf = [carry1_rf, rot_ny, rot_nx, carry2_rf] + pi_rf + [zero]
            name_rf = 'VQE_%02i_%02i_rf' % (i, j)
            sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
            sub_waves_rf.append(sub_rf)

            return sub, sub_waves, sub_rf, sub_waves_rf
        
        elif j == 12: # to check a different way to make N4 population
            #mw
            if int(t_2pi_rf_12_mwt/2) < 256:
                carry = Idle(0)
                carry.duration = int(t_2pi_rf_12_mwt/2)
                pulse_sequence = [carry] + Pi_24 + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rf_12_mwt/2) / 256, int(t_2pi_rf_12_mwt/2 ) % 256
                t_offset = n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence = [carry] + Pi_24 + [zero]
                wfm1 = IQWaveform('VQE_%02i_%02i_1'% (i, j), pulse_sequence, t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm1['I'])
                sub_waves.append(wfm1['Q'])
                sub.append(wfm1)
                sub_waves.append(sub)



            #rf
            pulse_sequence_rf = [zero] + pi_12_rf + [zero]
            name_rf = 'VQE_%02i_%02i_rf' % (i, j)
            sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
            sub_waves_rf.append(sub_rf)
            
                
            return sub, sub_waves, sub_rf, sub_waves_rf


        #elif j == 12:
            #mw
            #pulse_sequence = [zero] + Pi_all_x + [zero]
            #name = 'VQE_%02i_%02i' % (i, j)
            #sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
            #sub_waves.append(sub['I'])
            #sub_waves.append(sub['Q'])

            #rf
            #sub_rf = idle_rf
                
            #return sub, sub_waves, sub_rf, sub_waves_rf
        
        ### these 6 terms is used for checking N2 count normal or not.
        ''' 
        elif j == 12:   # check Npi_pi34_pi(N_2): without any pulse
            sub = idle_mw

            sub_rf = idle_rf

            return sub, sub_waves, sub_rf, sub_waves_rf
         
        elif j == 13:  # check Npi_pi34_pi(N_2): without rf pi_34_rf
            #mw
            pulse_sequence = [zero] + Pi_all_x + [zero]
            wfm1 = IQWaveform('VQE_%02i_%02i_1'% (i, j), pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
            sub_waves.append(wfm1['I'])
            sub_waves.append(wfm1['Q'])

            t_b, stub = wfm1['I'].duration, wfm1['I'].stub

            if int(t_2pi_rf_34_mwt/2) < stub:
                carry = Idle(0)
                carry.duration = int(t_2pi_rf_34_mwt/2)
                pulse_sequence = [zero] + Pi_all_x + [carry] + Pi_all_x + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )
                sub.append(wfm1)

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rf_34_mwt/2 - stub) / 256, int(t_2pi_rf_34_mwt/2 - stub) % 256
                t_offset = t_b + n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence = [carry] + Pi_all_x + [zero]
                wfm2 = IQWaveform('VQE_%02i_%02i_2'% (i, j), pulse_sequence, t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm2['I'])
                sub_waves.append(wfm2['Q'])
                sub.append(wfm2)
                sub_waves.append(sub)



            #rf
            sub_rf = idle_rf
                
            return sub, sub_waves, sub_rf, sub_waves_rf

  
        elif j == 14: # check Npi_pi34_pi(N_2): with a pi_34_rf_detune with freq 2.0 MHz
            #mw
            pulse_sequence = [zero] + Pi_all_x + [zero]
            wfm1 = IQWaveform('VQE_%02i_%02i_1'% (i, j), pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
            sub_waves.append(wfm1['I'])
            sub_waves.append(wfm1['Q'])

            t_b, stub = wfm1['I'].duration, wfm1['I'].stub

            if int(t_2pi_rf_34_mwt/2) < stub:
                carry = Idle(0)
                carry.duration = int(t_2pi_rf_34_mwt/2)
                pulse_sequence = [zero] + Pi_all_x + [carry] + Pi_all_x + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )
                sub.append(wfm1)

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rf_34_mwt/2 - stub) / 256, int(t_2pi_rf_34_mwt/2 - stub) % 256
                t_offset = t_b + n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence = [carry] + Pi_all_x + [zero]
                wfm2 = IQWaveform('VQE_%02i_%02i_2'% (i, j), pulse_sequence, t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm2['I'])
                sub_waves.append(wfm2['Q'])
                sub.append(wfm2)
                sub_waves.append(sub)



            #rf
            if int(t_2pi_rot_e_rft/2) < 512:
                carry = Idle(0)
                carry.duration = int(t_2pi_rot_e_rft/2) 

                pulse_sequence_rf = [carry] + pi_34_rf_detune + [zero]
                name_rf = 'VQE_%02i_%02i_rf' % (i, j)
                sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
                sub_waves_rf.append(sub_rf)
            else:# usually this won't happen
                sub_rf = Sequence('VQE_%02i_%02i_rf' % (i, j) )

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rot_e_rft/2) / 512, int(t_2pi_rot_e_rft/2) % 512
                
                if n > 0:
                    sub_rf.append(idle_rf, repeat=n)

                pulse_sequence_rf = [carry] + pi_34_rf_detune + [zero]
                name_rf = 'VQE_%02i_%02i_1_rf' % (i, j)
                wfm_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
                sub_waves_rf.append(wfm_rf)

                sub_rf.append(wfm_rf)
                sub_waves_rf.append(sub_rf)
                
            return sub, sub_waves, sub_rf, sub_waves_rf
            
  
        elif j == 15: # check Npi_pi34_pi(N_2): with a Big Pi(Pi_all_combined) consisting of pi_13 + pi_24
            #mw
            pulse_sequence = [zero] + Pi_all_combined + [zero]
            wfm1 = IQWaveform('VQE_%02i_%02i_1'% (i, j), pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
            sub_waves.append(wfm1['I'])
            sub_waves.append(wfm1['Q'])

            t_b, stub = wfm1['I'].duration, wfm1['I'].stub

            if int(t_2pi_rf_34_mwt/2) < stub:
                carry = Idle(0)
                carry.duration = int(t_2pi_rf_34_mwt/2)
                pulse_sequence = [zero] + Pi_all_combined + [carry] + Pi_all_combined + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )
                sub.append(wfm1)

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rf_34_mwt/2 - stub) / 256, int(t_2pi_rf_34_mwt/2 - stub) % 256
                t_offset = t_b + n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence = [carry] + Pi_all_combined + [zero]
                wfm2 = IQWaveform('VQE_%02i_%02i_2'% (i, j), pulse_sequence, t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm2['I'])
                sub_waves.append(wfm2['Q'])
                sub.append(wfm2)
                sub_waves.append(sub)



            #rf
            if int(t_2pi_mw_13_rft/2 + t_2pi_mw_24_rft/2) < 512:
                carry = Idle(0)
                carry.duration = int(t_2pi_mw_13_rft/2 + t_2pi_mw_24_rft/2) 

                pulse_sequence_rf = [carry] + pi_34_rf + [zero]
                name_rf = 'VQE_%02i_%02i_rf' % (i, j)
                sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
                sub_waves_rf.append(sub_rf)
            else:# usually this won't happen
                sub_rf = Sequence('VQE_%02i_%02i_rf' % (i, j) )

                carry = Idle(0)
                n, carry.duration = int(t_2pi_mw_13_rft/2 + t_2pi_mw_24_rft/2) / 512, int(t_2pi_mw_13_rft/2 + t_2pi_mw_24_rft/2) % 512
                
                if n > 0:
                    sub_rf.append(idle_rf, repeat=n)

                pulse_sequence_rf = [carry] + pi_34_rf + [zero]
                name_rf = 'VQE_%02i_%02i_1_rf' % (i, j)
                wfm_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
                sub_waves_rf.append(wfm_rf)

                sub_rf.append(wfm_rf)
                sub_waves_rf.append(sub_rf)
                
            return sub, sub_waves, sub_rf, sub_waves_rf
  
        elif j == 16:  # check Npi_pi34_pi(N_2): with a 2pi pulse of pi34 intead of pi
            #mw
            pulse_sequence = [zero] + Pi_all_x + [zero]
            wfm1 = IQWaveform('VQE_%02i_%02i_1'% (i, j), pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
            sub_waves.append(wfm1['I'])
            sub_waves.append(wfm1['Q'])

            t_b, stub = wfm1['I'].duration, wfm1['I'].stub

            if int(t_2pi_rf_34_mwt) < stub:
                carry = Idle(0)
                carry.duration = int(t_2pi_rf_34_mwt)
                pulse_sequence = [zero] + Pi_all_x + [carry] + Pi_all_x + [zero]
                name = 'VQE_%02i_%02i' % (i, j)
                sub = IQWaveform(name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)
                sub_waves.append(sub['I'])
                sub_waves.append(sub['Q'])
            else:
                sub = Sequence('VQE_%02i_%02i' % (i, j) )
                sub.append(wfm1)

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rf_34_mwt - stub) / 256, int(t_2pi_rf_34_mwt - stub) % 256
                t_offset = t_b + n * 256

                if n > 0:
                    sub.append(idle, idle, repeat=n)

                pulse_sequence = [carry] + Pi_all_x + [zero]
                wfm2 = IQWaveform('VQE_%02i_%02i_2'% (i, j), pulse_sequence, t_offset, file_type=0, sampling=self.sampling)
                sub_waves.append(wfm2['I'])
                sub_waves.append(wfm2['Q'])
                sub.append(wfm2)
                sub_waves.append(sub)



            #rf
            if int(t_2pi_rot_e_rft/2) < 512:
                carry = Idle(0)
                carry.duration = int(t_2pi_rot_e_rft/2) 

                pulse_sequence_rf = [carry] + pi_34_rf_x2 + [zero]
                name_rf = 'VQE_%02i_%02i_rf' % (i, j)
                sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
                sub_waves_rf.append(sub_rf)
            else:# usually this won't happen
                sub_rf = Sequence('VQE_%02i_%02i_rf' % (i, j) )

                carry = Idle(0)
                n, carry.duration = int(t_2pi_rot_e_rft/2) / 512, int(t_2pi_rot_e_rft/2) % 512
                
                if n > 0:
                    sub_rf.append(idle_rf, repeat=n)

                pulse_sequence_rf = [carry] + pi_34_rf_x2 + [zero]
                name_rf = 'VQE_%02i_%02i_1_rf' % (i, j)
                wfm_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
                sub_waves_rf.append(wfm_rf)

                sub_rf.append(wfm_rf)
                sub_waves_rf.append(sub_rf)
                
            return sub, sub_waves, sub_rf, sub_waves_rf
            
        elif j == 17:
            #mw
            sub = idle_mw

            #rf
            pulse_sequence_rf = [zero] + pi_12_rf + [zero]
            name_rf = 'VQE_%02i_%02i_rf' % (i, j)
            sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
            sub_waves_rf.append(sub_rf)

                
            return sub, sub_waves, sub_rf, sub_waves_rf
        '''    






        
    def compile_waveforms(self):

        waves = []
        waves_rf = []
        idle = Waveform('IDLE', [Idle(256)])
        waves.append(idle)
        idle_mw = IQWaveform('IDLE_mw', [Idle(256)])
        waves.append(idle_mw['I'])
        waves.append(idle_mw['Q'])
        idle_rf = Waveform610('IDLE_rf', [Idle(512)], sampling=self.sampling_rf)
        waves_rf.append(idle_rf)


        main_seq=Sequence('VQE')
        main_seq_rf=Sequence('VQE_rf')
        
        if self.check_state:
            i_range = 1
        else:
            i_range = 5

        for i in range(i_range):
            #for j in range(12 + self.test_n):
            for j in range(8):
                sub, sub_waves, sub_rf, sub_waves_rf = self.VQE_wfm(i,j)

                waves += sub_waves
                main_seq.append(sub, wait=True)
                waves_rf += sub_waves_rf
                main_seq_rf.append(sub_rf, wait=True)

        waves.append(main_seq)
        self.waves = waves
        self.main_wave = main_seq.name
        waves_rf.append(main_seq_rf)
        self.waves_rf = waves_rf
        self.main_wave_rf = main_seq_rf.name
    
    @on_trait_change('freq, freq_mw_13')
    def _update_freq_awg(self):
        self.freq_awg = self.freq_mw_13



    #==========================================================|
    #          treat raw data and store data in objects        |
    #==========================================================|
    @on_trait_change('count_data,integration_width,position_signal,position_normalize')
    def update_spin_state(self):
        y, profile, flank = spin_state(c=self.count_data,
                                       dt=self.bin_width,
                                       T=self.integration_width,
                                       t0=self.position_signal,
                                       t1=self.position_normalize,)

        y[y == np.inf] = 0 #turn all inf into 0
        y =  np.nan_to_num(y) #turn all NN into 0 

        self.spin_state = y
        self.spin_state_error = y ** 0.5
        self.pulse = profile
        self.flank = self.time_bins[flank]





    #==========================================================|
    #            create all the plots and container            |
    #==========================================================|
    def _create_matrix_plot(self):
        matrix_plot_data = ArrayPlotData(image=np.zeros((2, 2)))
        plot = Plot(matrix_plot_data, width=500, height=500, resizable='hv', padding=8, padding_left=48, padding_bottom=36)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'data index'
        plot.img_plot('image',
                      xbounds=(0, 1),
                      ybounds=(0, 1),
                      colormap=scheme['colormap']
                     )
        plot.tools.append(PanTool(plot))
        plot.overlays.append(SimpleZoom(plot, enable_wheel=False))
        self.matrix_plot_data = matrix_plot_data
        self.matrix_plot = plot

    def _create_pulse_plot(self):
        pulse_plot_data = ArrayPlotData(x=np.array((0., 0.1, 0.2)), y=np.array((0, 1, 2)))
        plot = Plot(pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('x', 'y'), style='line', color=scheme['data 1'], line_width=1, name='data')
        plot.bgcolor = scheme['background']
        plot.x_grid = None
        plot.y_grid = None
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'intensity [a.u.]'
        edge_marker = LinePlot(index = ArrayDataSource(np.array((0,0))),
                               value = ArrayDataSource(np.array((0,1e9))),
                               color = scheme['fit 1'],
                               index_mapper = LinearMapper(range=plot.index_range),
                               value_mapper = LinearMapper(range=plot.value_range),
                               name='marker')
        plot.add(edge_marker)
        plot.tools.append(PanTool(plot))
        plot.overlays.append(SimpleZoom(plot, enable_wheel=False))
        self.pulse_plot_data = pulse_plot_data
        self.pulse_plot = plot


    get_set_items = Pulsed.get_set_items + ['check_state', 'y1', 'spin_state', 'rot_n_y', 'rot_n_x', 'main_dir', 'main_dir_rf', 'amp1', 'amp1_rf', 'freq_rot_e', 'amp_rot_e', 't_2pi_rot_e', 'freq_rf_34', 'amp_rf_34', 't_2pi_rf_34', 'freq_mw_13', 'amp_mw_13', 't_2pi_mw_13', 'freq_mw_24', 'amp_mw_24', 't_2pi_mw_24', 'freq_rf_12', 'amp_rf_12', 't_2pi_rf_12']

    #==========================================================|
    #      update axis of the plots when relevant changes      |
    #==========================================================|

    @on_trait_change('time_bins, depth')
    def update_matrix_plot_axis(self):
        self.matrix_plot.components[0].index.set_data((self.time_bins[0], self.time_bins[-1]), (0.0, float(self.depth)))
    
    #==========================================================|
    #          update the plot data when count changes         |
    #==========================================================|
    @on_trait_change('count_data, show_raw_data')
    def update_matrix_plot(self):
        s = self.count_data.shape
        limit = 10000000000
        if (self.show_raw_data):
            if s[0] * s[1] < limit:
                self.matrix_plot_data.set_data('image', self.count_data)

    @on_trait_change('time_bins')
    def _update_pulse_index(self):
        self.pulse_plot_data.set_data('x', self.time_bins)

    @on_trait_change('pulse_profile')
    def _update_pulse_value(self):
        self.pulse_plot_data.set_data('y', self.pulse_profile)
    
    
    @on_trait_change('flank')
    def _on_flank_change(self,new):
       self.pulse_plot.components[1].index.set_data(np.array((new,new)))


    #==========================================================|
    #                   save data and graphs                   |
    #==========================================================|
    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)


    #==========================================================|
    #                   overwrite                              |
    #==========================================================|
    def update_fft(self):
        pass

    def _create_fft_plot(self):
        pass

    def _update_fft_plot_value(self):
        pass

    def _update_fft_axis_title(self):
        pass
    

    # line plot
    '''
    def _create_line_plot(self):
        line_plot_data = ArrayPlotData(index=np.array((0,1)), y1=np.array((0,0)), y2=np.array((0,0)))
        plot = Plot(line_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('index','y1'), color=scheme['data 1'], line_width=2, id='0', name='y1')
        plot.bgcolor = scheme['background']
        #plot.x_grid = None
        #plot.y_grid = None
        plot.index_axis.title = 'sequence'
        plot.value_axis.title = 'intensity [a.u.]'
        #plot.tools.append(PanTool(plot))
        #plot.overlays.append(SimpleZoom(plot, enable_wheel=True)) #changed to true
        self.line_plot_data = line_plot_data
        self.line_plot = plot

    
    @on_trait_change('y1')
    def _update_line_plot_value(self):
        y = self.y1/np.max(self.y1)
        n = len(y)
        old_index = self.line_plot_data.get_data('index')
        if old_index is not None and len(old_index) != n:
            self.line_plot_data.set_data('index', np.arange(n))
        self.line_plot_data.set_data('y1', y)

    
    
    @on_trait_change('count_data')
    def _update_line_plot_index(self):
        n = len(self.y1)
        self.line_plot_data.set_data('index', np.arange(n))
    '''




    def _create_line_plot(self):

        self.x_data = ArrayDataSource(np.arange(16))

        self.y1_data = ArrayDataSource(np.arange(16))

        self.index_range = DataRange1D(self.x_data, low_setting=-0.5, high_setting=7.5, stretch_data=False)
        self.index_mapper = LinearMapper(range=self.index_range)
        self.value_range = DataRange1D(self.y1_data, low_setting="auto", high_setting="auto", stretch_data=False)
        self.value_mapper = LinearMapper(range=self.value_range)
        self.plot1 = BarPlot(
			index=self.x_data, value=self.y1_data, value_mapper=self.value_mapper, index_mapper=self.index_mapper,
			alpha=0.5, line_color="black", orientation="h", fill_color=tuple(COLOR_PALETTE[5]), bar_width=1, antialias=False, padding=8, padding_left=64, padding_bottom=36
		)
        
        AXIS_DEFAULTS = {
			'axis_line_weight': 5,
			'tick_weight': 5,
			'tick_label_font': 'modern 12',
			'title_font': 'modern 16',
			'tick_out': 0,
			'tick_in': 10
		}
        _label = ["%d" % i for i in range(16)]
        _label_index = np.arange(16)
        self.x_axis = LabelAxis(labels=_label, positions=_label_index, component=self.plot1, orientation="bottom", ensure_labels_bounded=True, tick_label_position="outside", title="sequence", tight_bounds=True, **AXIS_DEFAULTS)
        self.y_axis = PlotAxis(mapper=self.value_mapper, component=self.plot1, orientation="left", ensure_labels_bounded=True, tick_label_position="outside", title="counts", tight_bounds=True, **AXIS_DEFAULTS)

        self.plot1.underlays.append(self.x_axis)
        self.plot1.underlays.append(self.y_axis)

    @on_trait_change("y1")
    def _update_line_plot_value(self):
        y = self.y1/np.max(self.y1)
        self.y1_data.set_data(y)
    

    
    @on_trait_change('count_data')
    def _update_line_plot_index(self):
        n = len(self.y1)
        self.x_data.set_data(np.arange(n))





    traits_view = View(
        VGroup(
            HGroup(
                Item('submit_button',   show_label=False),
                Item('remove_button',   show_label=False),
                Item('resubmit_button', show_label=False),
                Item('priority'),
                Item('state', style='readonly'),
                Item('run_time', style='readonly', format_str='%i'),
                Item('stop_time'),
            ),
            Tabbed(
                VGroup(
                    HGroup(
                        HGroup(
                            Item('freq', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                            Item('power', width=-40),
                            label='SMIQ',
                            show_border=True,
                        ),
                        HGroup(
                            Item('reload_awg'),
                            Item('reload_awg_rf'),
                            Item('wfm_button', show_label=False),
                            Item('wfm_button_rf', show_label=False),
                            Item('upload_progress', style='readonly', format_str='%i'),
                            Item('upload_progress_rf', style='readonly', format_str='%i'),
                            Item('main_dir', style='readonly'),
                            Item('main_dir_rf', style='readonly'),
                            Item('amp1', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80),
                            Item('amp1_rf', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80),
                            label='AWG',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item('check_state', width= -80, enabled_when='state != "run"'),
                        Item('rot_n_y', width= -80, enabled_when='state != "run"'),
                        Item('rot_n_x', width= -80, enabled_when='state != "run"'),
                        label='VQE parameters',
                        show_border=True,
                    ),
                    HGroup(
                        Item('freq_rot_e', width= -80, enabled_when='state != "run"'),
                        Item('amp_rot_e', width= -80, enabled_when='state != "run"'),
                        Item('t_2pi_rot_e', width= -80, enabled_when='state != "run"'),
                        Item('freq_rf_34', width= -80, enabled_when='state != "run"'),
                        Item('amp_rf_34', width= -80, enabled_when='state != "run"'),
                        Item('t_2pi_rf_34', width= -80, enabled_when='state != "run"'),
                        Item('freq_rf_12', width= -80, enabled_when='state != "run"'),
                        Item('amp_rf_12', width= -80, enabled_when='state != "run"'),
                        Item('t_2pi_rf_12', width= -80, enabled_when='state != "run"'),
                        label='Unconditional MW Gate & rf_34 gate',
                        show_border=True,
                    ),
                    HGroup(
                        Item('freq_mw_13', width= -80, enabled_when='state != "run"'),
                        Item('amp_mw_13', width= -80, enabled_when='state != "run"'),
                        Item('t_2pi_mw_13', width= -80, enabled_when='state != "run"'),
                        Item('freq_mw_24', width= -80, enabled_when='state != "run"'),
                        Item('amp_mw_24', width= -80, enabled_when='state != "run"'),
                        Item('t_2pi_mw_24', width= -80, enabled_when='state != "run"'),
                        label='Conditional MW Gate',
                        show_border=True,
                    ),
                    VGroup(
                        Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=580, height=250, resizable=True),
                        Item('plot1', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                    ),
                    label = 'data',
                ),
                VGroup(
                    HGroup(
                        HGroup(
                            Item('laser', width=-50),
                            Item('wait', width=-50),
                            label='Sequence',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        HGroup(
                            Item('vpp1', width=-50),
                            Item('vpp2', width=-50),
                            Item('sampling', width=-80),
                            Item('trig_interval', width=-80),
                            Item('trig_delay', width=-80),
                            label='AWG',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        HGroup(
                            Item('vpp_rf', width=-50),
                            Item('sampling_rf', width=-80),
                            Item('trig_interval_rf', width=-80),
                            Item('trig_delay_rf', width=-80),
                            label='AWG_rf',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item('record_length'),
                        Item('bin_width'),
                        Item('time_window_width'),
                        Item('time_window_offset_signal'),
                        Item('time_window_offset_normalize'),
                        Item('dual', enabled_when='state != "run"'),
                        label='Analysis',
                        show_border=True,
                    ),
                    Item('pulse_plot', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                    label = 'settings',
                ),
            ),
        ),
        title='VQE en',
        buttons=[],
        resizable=True,
        width =1500,
        height=800,
        handler=VQEHandler,
        menubar=MenuBar(
            Menu(
                Action(action='load', name='Load'),
                Action(action='save', name='Save (.pyd or .pys)'),
                Action(action='saveMatrixPlot', name='SaveMatrixPlot (.png)'),
                Action(action='saveAll', name='Save All (.png+.pys)'),
                Action(action='_on_close', name='Quit'),
                name='File',
            ),
        ),
    )
    
