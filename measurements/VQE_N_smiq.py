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

from measurements.pulsed_awg_rf import Pulsed



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


class VQE_N_smiq(Pulsed):
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

    #check_state = Bool(False, desc='only check state, no gradient', label='check state')

    frequency  = Range(low=1, high=20e9,  value=2.70e9, desc='SMIQ frequency', label='SMIQ frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))


    #awg parameters
    #main_dir = String('\VQE', desc='.WFM/.PAT folder in AWG 520', label='Dir.')
    main_dir_rf = String('\VQE', desc='.WFM/.PAT folder in AWG 610', label='rf Dir.')
    #amp1  = Range(low=0.0, high=1.0, value=1.0, desc='Waveform amplitude factor', label='wfm amp', auto_set=False, enter_set=True)
    amp1_rf  = Range(low=0.0, high=1.0, value=1.0, desc='Waveform_rf amplitude factor', label='wfm rf amp', auto_set=False, enter_set=True)

    '''
    #variational quamtum eigensolver
    rot_e_y = Range(low=0., high=1., value=0., desc='rotation of electron spin about y axis', label='Rotation y eletron spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)
    rot_e_x = Range(low=0., high=1., value=0., desc='rotation of electron spin about x axis', label='Rotation x eletron spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)
    rot_n_y = Range(low=0., high=1., value=0., desc='rotation of nuclear spin about y axis', label='Rotation y nuclear spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)
    rot_n_x = Range(low=0., high=1., value=0., desc='rotation of nuclear spin about x axis', label='Rotation x nuclear spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)

    freq_rot_e = Range(low=1, high=1e9, value=48.92e6, desc='microwave frequency for rot e', label='frequency rot_e [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    amp_rot_e = Range(low= 0., high=1., value= 1, desc='amp for rot e', label='amp rot_e[500mVpp]', mode='text', auto_set=False, enter_set=True)
    '''
    t_2pi_rot_e = Range(low=1., high=100000., value=200., desc='rabi period 2pi pulse length for for rot e', label='2pi t rot_e[ns]', mode='text', auto_set=False, enter_set=True)
    

    
    #Qubit controlling parameters
    #The 4 energy levels is defined 3(electron spin:up, nuclear spin:up) 4(up, down) 1(down, up) 2(down, down)
    freq_rf_34 = Range(low=1, high=2.6e9, value=2.940e+6, desc='radio frequency of energy gap between 3 and 4', label='frequency rf_34 [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    amp_rf_34 = Range(low= 0., high=1., value= 1, desc='amp of radiowave of energy gap between 3 and 4', label='amp rf_34[500mVpp]', mode='text', auto_set=False, enter_set=True)
    t_2pi_rf_34 = Range(low=1., high=5000000., value=200.e3, desc='rabi period 2pi pulse length for rf_34', label='2pi t rf_34[ns]', mode='text', auto_set=False, enter_set=True)
    
    '''
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
    '''


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



    def __init__(self, pulse_generator, time_tagger, microwave, awg_rf, **kwargs):
        self.generate_sequence()
        
        super(VQE_N_smiq, self).__init__(pulse_generator, time_tagger, microwave, awg_rf, **kwargs)
        
        
        #create different plots
        self._create_matrix_plot()
        self._create_pulse_plot()
        self._create_line_plot()




    def generate_sequence(self):
        laser = self.laser
        wait  = self.wait
        #trig_interval = self.trig_interval
        #trig_delay = self.trig_delay
        trig_interval_rf = self.trig_interval_rf
        trig_delay_rf = self.trig_delay_rf



        t_2pi_rot_e = self.t_2pi_rot_e
        t_2pi_rf_34 = self.t_2pi_rf_34

        '''
        # 4 kinds of sequence for normlize N1 N3 N4 N2
        norm_sequences = []
        norm_sequences.append(([], 0))   # N1    
        norm_sequences.append(([], t_2pi_rot_e/2)) # N3  
        norm_sequences.append(([], t_2pi_rot_e/2 + t_2pi_rf_34/2)) # N4  
        norm_sequences.append(([], t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2)) # N2
        '''        

        max_time = t_2pi_rot_e/2 + t_2pi_rf_34/2 + t_2pi_rot_e/2 + trig_interval_rf + trig_delay_rf

        sequence = [
            (['aom'], laser),
            (['aom'], (576) * 1e9 /self.sampling_rf + 450 ),#to be the same with other laser
            ([ ], wait )
        ]
        
        
        sequence.append((['ch2'], 50)) #consider (['sequence'], 100)
        sequence.append((['B'], max_time)) #consider (['sequence'], 100)
        sequence.append((['B','aom', 'laser'], laser))
        sequence.append((['B','aom' ], (576) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
        sequence.append((['B'], wait))

        
        sequence.append((['B'], max_time - t_2pi_rot_e/2)) #consider (['sequence'], 100)
        sequence.append((['B','A','mw_x'], t_2pi_rot_e/2))
        sequence.append((['B','laser', 'aom'], laser))
        sequence.append((['B','aom' ], (576) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
        sequence.append((['B'  ], wait))


        
        sequence.append((['B'], max_time - t_2pi_rf_34/2 - trig_interval_rf - trig_delay_rf - t_2pi_rot_e/2)) #consider (['sequence'], 100)
        sequence.append((['B','awg_rf'],  trig_interval_rf))
        sequence.append((['B'], trig_delay_rf))
        sequence.append((['B','A','mw_x'], t_2pi_rot_e/2))
        sequence.append((['B'], t_2pi_rf_34/2))
        sequence.append((['B','laser', 'aom'], laser))
        sequence.append((['B','aom' ], (576) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
        sequence.append((['B'], wait))


        
        sequence.append((['B'], max_time - t_2pi_rf_34/2 - trig_interval_rf - trig_delay_rf - t_2pi_rot_e/2 - t_2pi_rot_e/2)) #consider (['sequence'], 100)
        sequence.append((['B','awg_rf'],  trig_interval_rf))
        sequence.append((['B'], trig_delay_rf))
        sequence.append((['B','A','mw_x'], t_2pi_rot_e/2))
        sequence.append((['B'], t_2pi_rf_34/2))
        sequence.append((['B','A','mw_x'], t_2pi_rot_e/2))
        sequence.append((['B','laser', 'aom'], laser))
        sequence.append((['B','aom' ], (576) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
        sequence.append((['B'], wait))
        



        sequence.append((['sequence'], 100))
        return sequence




        
    def compile_waveforms(self):

        #waves = []
        waves_rf = []
        #idle = Waveform('IDLE', [Idle(256)])
        #waves.append(idle)
        #idle_mw = IQWaveform('IDLE_mw', [Idle(256)])
        #waves.append(idle_mw['I'])
        #waves.append(idle_mw['Q'])
        idle_rf = Waveform610('IDLE_rf', [Idle(512)], sampling=self.sampling_rf)
        waves_rf.append(idle_rf)


        #main_seq=Sequence('VQE')
        main_seq_rf=Sequence('VQE_rf')
        
        '''
        if self.check_state:
            i_range = 1
        else:
            i_range = 5
        '''

        t_2pi_rf_34 = self.t_2pi_rf_34 * self.sampling_rf / 1e9
        t_2pi_rot_e_rft = self.t_2pi_rot_e * self.sampling_rf / 1e9
        freq_rf_34 = self.freq_rf_34/self.sampling_rf

        zero = Idle(1)
        pi_34_rf = [Sin(t_2pi_rf_34/2, freq=freq_rf_34, amp=self.amp_rf_34, phase= 0)]

        carry = Idle(0)
        carry.duration = int(t_2pi_rot_e_rft/2) 

        sub_waves_rf = []
        pulse_sequence_rf = [carry] + pi_34_rf + [zero]
        name_rf = 'VQE_rf_pi' 
        sub_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)
        sub_waves_rf.append(sub_rf)


        waves_rf += sub_waves_rf
        main_seq_rf.append(sub_rf, wait=True)

        #waves.append(main_seq)
        #self.waves = waves
        #self.main_wave = main_seq.name
        waves_rf.append(main_seq_rf)
        self.waves_rf = waves_rf
        self.main_wave_rf = main_seq_rf.name
    




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


    get_set_items = Pulsed.get_set_items + [ 'y1', 'spin_state', 'main_dir_rf', 'amp1_rf',  't_2pi_rot_e', 'freq_rf_34', 'amp_rf_34', 't_2pi_rf_34']

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

        self.index_range = DataRange1D(self.x_data, low_setting=-0.5, high_setting=3.5, stretch_data=False)
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
                            Item('frequency', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                            Item('power', width=-40),
                            Item('t_2pi_rot_e', width= -80, enabled_when='state != "run"'),
                            label='SMIQ',
                            show_border=True,
                        ),
                        HGroup(
                            Item('reload_awg_rf'),
                            Item('wfm_button_rf', show_label=False),
                            Item('upload_progress_rf', style='readonly', format_str='%i'),
                            Item('main_dir_rf', style='readonly'),
                            Item('amp1_rf', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80),
                            label='AWG_rf',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item('freq_rf_34', width= -80, enabled_when='state != "run"'),
                        Item('amp_rf_34', width= -80, enabled_when='state != "run"'),
                        Item('t_2pi_rf_34', width= -80, enabled_when='state != "run"'),
                        label='rf',
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
        title='VQE N smiq',
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
    
