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
from tools.cron import CronDaemon, CronEvent

from measurements.pulsed_3smiq_awg_rf import Pulsed



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


class VQE_2p(Pulsed):
    '''
    +======================================================================================================+
    |                                                                                                      
    |           VARIATIONAL QUANTUM EIGENSOLVER with tomography                                            
    |                                                                                                      
    |           Laser + rf_t(theta) + rf_x(theta) + mw_t(theta) + mw_x(theta) + TOMO             
    |                                                                                                      
    |           normalization: 4 sequence                       
    |           theta : 1 for this point                                   
    |           Tomo : 4 for diag matrix elements; 4 for off diag terms                                          
    |                   
    |
    |
    |           Devices: 3smiq for MW (mw_13, mw_24 CNot gate and mw for single qubit gate)
    |                    AWG610 for RF
    |
    |           The 4 energy levels is defined 1 (e = 0, n = +1)     2 (0,0)     3 (-1,+1)    4 (-1,0)
    |
    |                                                                                                     
    | Ref: Decoherence-protected quantum gates for a hybrid solid-state spin register                     
    +======================================================================================================+
    '''

    '''
    version:
    2023/05/24:
                big pi changed back to hard pi instead of combined pi
    2023/05/22:
                range the rf duration between -0.5~0.5 instead of 0~1 to shorten the rf duration and speed up the measurement
    2023/05/19:
                considered phase accumulated during single qubit electron spin gate
    '''

    '''
    ToDo:

    '''


    'auto save'
    auto_save = Bool(False,label='auto save')
    state_auto_save = Enum('idle', 'stopped', 'auto saving', 'error')
    saveWaitTime = Range(low= 1, high=1440, value= 60, desc='time interval of auto save', label='save every [min]', mode='text', auto_set=False, enter_set=True)
    auto_save_dir = String('D:/Data/2023/VQE_default', desc='auto save dir', label='auto save dir', mode='text')
    #auto_save_dir = 'D:/Data/2023/VQE_data_20230323/VQE_0.375_0.125/'
    save_count_name = 0
    
    one_round_time = Range(low= 1, high=1e9, value= 60, desc='sequence length for one round', label='sequence length[ns]', mode='text', auto_set=False, enter_set=True)

    splitting = Range(low=1, high=20e9,  value=2.160241e6, desc='peak splitting', label='splitting [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))


    #awg parameters
    main_dir_rf = String('\VQE', desc='.WFM/.PAT folder in AWG 610', label='rf Dir.')
    amp1_rf  = Range(low=0.0, high=1.0, value=1.0, desc='Waveform_rf amplitude factor', label='wfm rf amp', auto_set=False, enter_set=True)

    
    #variational quamtum eigensolver
    rot_e_y = Range(low=0., high=1., value=0., desc='rotation of electron spin about y axis', label='Rotation y eletron spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)
    rot_e_x = Range(low=0., high=1., value=0., desc='rotation of electron spin about x axis', label='Rotation x eletron spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)
    rot_n_y = Range(low=-0.5, high=0.5, value=0., desc='rotation of nuclear spin about y axis', label='Rotation y nuclear spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)
    rot_n_x = Range(low=-0.5, high=0.5, value=0., desc='rotation of nuclear spin about x axis', label='Rotation x nuclear spin(*2 Pi)', mode='text', auto_set=False, enter_set=True)

    mw_wait = Range(low=1., high=5000000., value=400, desc='wait time after rf before mw', label='mw wait[ns]', mode='text', auto_set=False, enter_set=True)

    phase_23 = Range(low=0., high=10., value=0.0, desc='the phase change for rho23 term measurement', label='Phase 23', mode='text', auto_set=False, enter_set=True)
    phase_14 = Range(low=0., high=10., value=0.0, desc='the phase change for rho14 term measurement', label='Phase 14', mode='text', auto_set=False, enter_set=True)

    
    #Qubit controlling parameters
    freq_rf_34 = Range(low=1, high=2.6e9, value=2.940e+6, desc='radio frequency of energy gap between 3 and 4', label='frequency rf_34 [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    amp_rf_34 = Range(low= 0., high=1., value= 1, desc='amp of radiowave of energy gap between 3 and 4', label='amp rf_34[500mVpp]', mode='text', auto_set=False, enter_set=True)
    t_2pi_rf_34 = Range(low=1., high=5000000., value=200.e3, desc='rabi period 2pi pulse length for rf_34', label='2pi t rf_34[ns]', mode='text', auto_set=False, enter_set=True)

    t_2pi_mw_all_y = Range(low=1., high=100000., value=1000., desc='rabi period 2pi pulse length for mw_all_y', label='2pi t mw_all_y[ns]', mode='text', auto_set=False, enter_set=True)

    t_2pi_mw_all_x = Range(low=1., high=100000., value=1000., desc='rabi period 2pi pulse length for mw_all_x', label='2pi t mw_all_x[ns]', mode='text', auto_set=False, enter_set=True)
    
    freq_rf_12 = Range(low=1, high=2.6e9, value=2.940e+6, desc='radio frequency of energy gap between 1 and 2', label='frequency rf_12 [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    amp_rf_12 = Range(low= 0., high=1., value= 1, desc='amp of radiowave of energy gap between 1 and 2', label='amp rf_12[500mVpp]', mode='text', auto_set=False, enter_set=True)
    t_2pi_rf_12 = Range(low=1., high=5000000., value=200.e3, desc='rabi period 2pi pulse length for rf_12', label='2pi t rf_12[ns]', mode='text', auto_set=False, enter_set=True)


    t_2pi_mw_13 = Range(low=1., high=100000., value=1000., desc='rabi period 2pi pulse length for mw_13', label='2pi t mw_13[ns]', mode='text', auto_set=False, enter_set=True)


    t_2pi_mw_24 = Range(low=1., high=100000., value=1000., desc='rabi period 2pi pulse length for mw_24', label='2pi t mw_24[ns]', mode='text', auto_set=False, enter_set=True)




    """heating"""
    freq_awg_rf_heating = Range(low=1, high=2.6e9, value=2.5e6, desc='AWG_rf frequency heating', label='freq heating[Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    


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


    heating_time   = Instance( list, factory=list )

    def __init__(self, pulse_generator, time_tagger, microwave, microwave2, microwave3, awg_rf, **kwargs):
        self.generate_sequence()
        
        super(VQE_2p, self).__init__(pulse_generator, time_tagger, microwave, microwave2, microwave3, awg_rf, **kwargs)
        
        
        #create different plots
        self._create_matrix_plot()
        self._create_pulse_plot()
        self._create_line_plot()
        self.frequency2 = self.frequency - self.splitting
        self.frequency3 = self.frequency - self.splitting/2.0
        
    
    def submit(self):
        """Start new measurement without keeping data."""
        self.keep_data = False
        self.save_count_name = 0
        ManagedJob.submit(self)

    def auto_save_func(self):
        '''run every checkWaitTime'''
        try:
            if (self.auto_save == True):
                if (self.state == 'run'):
                    self.save_count_name = self.save_count_name + 1
                    #saving files
                    #dir1 = 'vqe_%03.3f_%03.3f_%i.pyd' % (self.rot_n_y, self.rot_n_x, self.save_count_name)
                    #self.save(self.auto_save_dir + dir1)
                    dir2 = '/vqe_%03.3f_%03.3f_%03.3f_%03.3f_%imin_%i.pys' % (self.rot_n_y, self.rot_n_x, self.rot_e_y, self.rot_e_x, self.saveWaitTime, self.save_count_name)
                    dir_save = self.auto_save_dir + dir2
                    print(dir_save)
                    self.save(dir_save)
            else:
                pass
        except:
            self.auto_save = False
            self.state_auto_save = "error"
            logging.getLogger().exception('There was an Error.')
            print('There was an Error in auto save.')
            
    def _auto_save_changed(self, new):
        if not new and hasattr(self, 'cron_event'):
            CronDaemon().remove(self.cron_event)
            self.state_auto_save= 'stopped'
        if new:
            self.cron_event = CronEvent(self.auto_save_func, min=range(0,1440,self.saveWaitTime))
            CronDaemon().register(self.cron_event)
            self.state_auto_save= 'auto saving'

    
    @on_trait_change('frequency')
    def _update_frequency2(self):
        self.frequency2 = self.frequency - self.splitting

    @on_trait_change('frequency')
    def _update_frequency3(self):
        self.frequency3 = self.frequency - self.splitting/2.0

    'over write to add generate heating_time'
    def load_wfm_rf(self):
        self.tau = np.arange(self.start_time, self.end_time, self.time_step)

        self.waves_rf = []
        self.main_wave_rf = ''

        self.generate_heating_time()
        self.compile_waveforms()

        self.awg_rf.ftp_cwd = '/main' + self.main_dir_rf
        self.awg_rf.upload(self.waves_rf)
        self.awg_rf.managed_load(self.main_wave_rf,cwd=self.main_dir_rf)
        #self.awg.managed_load(self.main_wave)
        self.reload_awg_rf = False


    def generate_heating_time(self):

        t_2pi_rf_34 = self.t_2pi_rf_34
        t_2pi_rf_12 = self.t_2pi_rf_12
        t_2pi_mw_all_y = self.t_2pi_mw_all_y
        t_2pi_mw_all_x = self.t_2pi_mw_all_x

        t_2pi_mw_13 = self.t_2pi_mw_13
        t_2pi_mw_24 = self.t_2pi_mw_24

        mw_pi = t_2pi_mw_13/2
        mw_pi_2 = t_2pi_mw_24/2
        mw_pi_rf = t_2pi_rf_34/2
        mw_pi_rf_12 = t_2pi_rf_12/2

        mw_pi2_rf = t_2pi_rf_34/4

        rot_n_y = abs(self.rot_n_y)
        rot_n_x = abs(self.rot_n_x)
        rot_e_y = self.rot_e_y
        rot_e_x = self.rot_e_x

        mw_wait = self.mw_wait

        t_rot_rf = (rot_n_y + rot_n_x) * t_2pi_rf_34 
        t_rot_e_y = rot_e_y  * t_2pi_mw_all_y
        t_rot_e_x = rot_e_x  * t_2pi_mw_all_x
        

        hard_pi_2_t = 0.5 * t_2pi_mw_all_x

        if t_rot_e_x > hard_pi_2_t:
            t_rot_e_x_23 = t_rot_e_x - hard_pi_2_t
        else:
            t_rot_e_x_23 = t_rot_e_x + hard_pi_2_t



        '''
        if mw_pi > mw_pi_2:
            hard_pi_seq = [
                (['mw_13','mw_24'], mw_pi_2),
                (['mw_13'], mw_pi - mw_pi_2)
            ]
            hard_pi_t = mw_pi
        else:
            hard_pi_seq = [
                (['mw_13','mw_24'], mw_pi),
                (['mw_24'], mw_pi_2 - mw_pi)
            ]
            hard_pi_t = mw_pi_2
        '''
        
        hard_pi_t = hard_pi_2_t


        """max_time is defined as the max time interval after trig_delay_rf and before laser"""
        max_time = max(hard_pi_t + t_rot_rf + mw_wait + t_rot_e_y + t_rot_e_x + mw_wait + mw_pi + mw_pi_rf , mw_pi_rf_12, hard_pi_t + t_rot_rf + mw_wait + t_rot_e_y + t_rot_e_x + mw_wait + mw_pi2_rf + mw_wait + mw_pi + mw_wait, hard_pi_t + t_rot_rf + mw_wait + t_rot_e_y + t_rot_e_x_23 + mw_wait + mw_pi2_rf + mw_wait + mw_pi + mw_wait)


        heating_time = []


        'N1'
        heating_time.append(max_time)

        'N2'
        heating_time.append(max_time - mw_pi_rf_12)
        '''
        sequence += hard_pi_seq
        sequence.append(([], mw_pi_rf))
        sequence += hard_pi_seq
        sequence.append(([], max_time?))
        '''

        'N3'
        heating_time.append(max_time - hard_pi_t)

        'N4'
        heating_time.append(max_time - hard_pi_t - mw_pi_rf)

        "Every t need to measure 4 kinds of population"
        'Tomo 0'
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_pi - mw_wait)

        'Tomo pi13'
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_wait)

        'Tomo pi34'
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_pi - mw_wait - mw_pi_rf)

        'Tomo Pi'
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_pi_2 - mw_wait)


        "Non-diag terms"
        'rho23 +pi/2'
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x_23 - mw_wait - mw_pi2_rf - mw_wait - mw_pi - mw_wait)


        'rho23 -pi/2'
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x_23 - mw_wait - mw_pi2_rf - mw_wait - mw_pi - mw_wait)


        'rho14 +pi/2'
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_wait - mw_pi2_rf - mw_wait - mw_pi - mw_wait)


        'rho14 -pi/2'
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_wait - mw_pi2_rf - mw_wait - mw_pi - mw_wait)

  
        self.heating_time = heating_time
        return heating_time


    # Sequence ############################
    def generate_sequence(self):
        laser = self.laser
        wait  = self.wait
        trig_interval_rf = self.trig_interval_rf
        trig_delay_rf = self.trig_delay_rf
        

        t_2pi_rf_34 = self.t_2pi_rf_34
        t_2pi_rf_12 = self.t_2pi_rf_12
        t_2pi_mw_all_y = self.t_2pi_mw_all_y
        t_2pi_mw_all_x = self.t_2pi_mw_all_x

        t_2pi_mw_13 = self.t_2pi_mw_13
        t_2pi_mw_24 = self.t_2pi_mw_24

        mw_pi = t_2pi_mw_13/2
        mw_pi_2 = t_2pi_mw_24/2
        mw_pi_rf = t_2pi_rf_34/2
        mw_pi_rf_12 = t_2pi_rf_12/2

        mw_pi2_rf = t_2pi_rf_34/4

        
        rot_n_y = abs(self.rot_n_y)
        rot_n_x = abs(self.rot_n_x)
        rot_e_y = self.rot_e_y
        rot_e_x = self.rot_e_x

        mw_wait = self.mw_wait

        t_rot_rf = (rot_n_y + rot_n_x) * t_2pi_rf_34 
        t_rot_e_y = rot_e_y  * t_2pi_mw_all_y
        t_rot_e_x = rot_e_x  * t_2pi_mw_all_x


        hard_pi_2_t = 0.5 * t_2pi_mw_all_x

        if t_rot_e_x > hard_pi_2_t:
            t_rot_e_x_23 = t_rot_e_x - hard_pi_2_t
        else:
            t_rot_e_x_23 = t_rot_e_x + hard_pi_2_t

        '''
        if mw_pi > mw_pi_2:
            hard_pi_seq = [
                (['mw_13','mw_24'], mw_pi_2),
                (['mw_13'], mw_pi - mw_pi_2)
            ]
            hard_pi_t = mw_pi
        else:
            hard_pi_seq = [
                (['mw_13','mw_24'], mw_pi),
                (['mw_24'], mw_pi_2 - mw_pi)
            ]
            hard_pi_t = mw_pi_2
        '''
        
        hard_pi_seq = [
            (['mw_all_x'], hard_pi_2_t),
        ]
        hard_pi_t = hard_pi_2_t

        """max_time is defined as the max time interval after trig_delay_rf and before laser"""
        max_time = max(hard_pi_t + t_rot_rf + mw_wait + t_rot_e_y + t_rot_e_x + mw_wait + mw_pi + mw_pi_rf , mw_pi_rf_12, hard_pi_t + t_rot_rf + mw_wait + t_rot_e_y + t_rot_e_x + mw_wait + mw_pi2_rf + mw_wait + mw_pi + mw_wait, hard_pi_t + t_rot_rf + mw_wait + t_rot_e_y + t_rot_e_x_23 + mw_wait + mw_pi2_rf + mw_wait + mw_pi + mw_wait)


        heating_time = []

        """To use the 1st smiq, generate 'mw_13' trigger"""
        """To use the 2nd smiq, generate 'mw_24' trigger"""
        """To use the 3rd smiq, generate 'mw_all_y' or 'mw_all_x' trigger"""
        
      
        #region

        norm_sequences = [[],[],[],[]]

        'N1'
        heating_time.append(max_time)

        'N2'
        norm_sequences[1].append(([], mw_pi_rf_12))
        heating_time.append(max_time - mw_pi_rf_12)
        '''
        sequence += hard_pi_seq
        sequence.append(([], mw_pi_rf))
        sequence += hard_pi_seq
        sequence.append(([], max_time?))
        '''

        'N3'
        norm_sequences[2] += hard_pi_seq
        heating_time.append(max_time - hard_pi_t)

        'N4'
        norm_sequences[3] += hard_pi_seq
        norm_sequences[3].append(([], mw_pi_rf))
        heating_time.append(max_time - hard_pi_t - mw_pi_rf)



        "Every t need to measure 4 kinds of population"
        diag_sequences = [[],[],[],[]]

        'Tomo 0'
        diag_sequences[0] += hard_pi_seq
        diag_sequences[0].append(([], t_rot_rf))
        diag_sequences[0].append(([], mw_wait))
        diag_sequences[0].append((['mw_all_y'], t_rot_e_y))
        diag_sequences[0].append((['mw_all_x'], t_rot_e_x))
        diag_sequences[0].append((['mw_13'], mw_pi))
        diag_sequences[0].append(([], mw_wait))
        #diag_sequences[0].append(([], max_time - hard_pi_t - t_rot_rf - mw_pi)) 
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_pi - mw_wait)

        'Tomo pi13'
        diag_sequences[1] += hard_pi_seq
        diag_sequences[1].append(([], t_rot_rf))
        diag_sequences[1].append(([], mw_wait))
        diag_sequences[1].append((['mw_all_y'], t_rot_e_y))
        diag_sequences[1].append((['mw_all_x'], t_rot_e_x))
        diag_sequences[1].append(([], mw_wait))
        #diag_sequences[1].append(([], max_time - hard_pi_t - t_rot_rf)) 
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_wait)

        'Tomo pi34'
        diag_sequences[2] += hard_pi_seq
        diag_sequences[2].append(([], t_rot_rf))
        diag_sequences[2].append(([], mw_wait))
        diag_sequences[2].append((['mw_all_y'], t_rot_e_y))
        diag_sequences[2].append((['mw_all_x'], t_rot_e_x))
        diag_sequences[2].append((['mw_13'], mw_pi))
        diag_sequences[2].append(([], mw_wait))
        diag_sequences[2].append(([], mw_pi_rf))
        #diag_sequences[2].append(([], max_time - hard_pi_t - t_rot_rf - mw_pi - mw_pi_rf)) 
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_pi - mw_wait - mw_pi_rf)

        'Tomo Pi'
        diag_sequences[3] += hard_pi_seq
        diag_sequences[3].append(([], t_rot_rf))
        diag_sequences[3].append(([], mw_wait))
        diag_sequences[3].append((['mw_all_y'], t_rot_e_y))
        diag_sequences[3].append((['mw_all_x'], t_rot_e_x))
        diag_sequences[3].append((['mw_24'], mw_pi_2))
        diag_sequences[3].append(([], mw_wait))
        #diag_sequences[3].append(([], max_time - hard_pi_t - t_rot_rf - mw_pi_2)) 
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_pi_2 - mw_wait)

        #endregion


        "Non-diag terms. 2 for Real rho23. 2 for Real rho14"
        nondiag_sequences = [[],[],[],[]]

        'rho23 +pi/2'
        nondiag_sequences[0] += hard_pi_seq
        nondiag_sequences[0].append(([], t_rot_rf))
        nondiag_sequences[0].append(([], mw_wait))
        nondiag_sequences[0].append((['mw_all_y'], t_rot_e_y))
        nondiag_sequences[0].append((['mw_all_x'], t_rot_e_x_23))
        nondiag_sequences[0].append(([], mw_wait))
        nondiag_sequences[0].append(([], mw_pi2_rf))
        nondiag_sequences[0].append(([], mw_wait))
        nondiag_sequences[0].append((['mw_13'], mw_pi))
        nondiag_sequences[0].append(([], mw_wait))
        #nondiag_sequences[0].append(([], max_time - hard_pi_t - t_rot_rf - hard_pi_t - mw_pi2_rf - mw_pi_2)) 
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x_23 - mw_wait - mw_pi2_rf - mw_wait - mw_pi - mw_wait)


        'rho23 -pi/2'
        nondiag_sequences[1] += hard_pi_seq
        nondiag_sequences[1].append(([], t_rot_rf))
        nondiag_sequences[1].append(([], mw_wait))
        nondiag_sequences[1].append((['mw_all_y'], t_rot_e_y))
        nondiag_sequences[1].append((['mw_all_x'], t_rot_e_x_23))
        nondiag_sequences[1].append(([], mw_wait))
        nondiag_sequences[1].append(([], mw_pi2_rf))
        nondiag_sequences[1].append(([], mw_wait))
        nondiag_sequences[1].append((['mw_13'], mw_pi))
        nondiag_sequences[1].append(([], mw_wait))
        #nondiag_sequences[1].append(([], max_time - hard_pi_t - t_rot_rf - hard_pi_t - mw_pi2_rf - mw_pi_2)) 
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x_23 - mw_wait - mw_pi2_rf - mw_wait - mw_pi - mw_wait)


        'rho14 +pi/2'
        nondiag_sequences[2] += hard_pi_seq
        nondiag_sequences[2].append(([], t_rot_rf))
        nondiag_sequences[2].append(([], mw_wait))
        nondiag_sequences[2].append((['mw_all_y'], t_rot_e_y))
        nondiag_sequences[2].append((['mw_all_x'], t_rot_e_x))
        nondiag_sequences[2].append(([], mw_wait))
        nondiag_sequences[2].append(([], mw_pi2_rf))
        nondiag_sequences[2].append(([], mw_wait))
        nondiag_sequences[2].append((['mw_13'], mw_pi))
        nondiag_sequences[2].append(([], mw_wait))
        #nondiag_sequences[2].append(([], max_time - hard_pi_t - t_rot_rf - mw_pi2_rf - mw_pi)) 
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_wait - mw_pi2_rf - mw_wait - mw_pi - mw_wait)


        'rho14 -pi/2'
        nondiag_sequences[3] += hard_pi_seq
        nondiag_sequences[3].append(([], t_rot_rf))
        nondiag_sequences[3].append(([], mw_wait))
        nondiag_sequences[3].append((['mw_all_y'], t_rot_e_y))
        nondiag_sequences[3].append((['mw_all_x'], t_rot_e_x))
        nondiag_sequences[3].append(([], mw_wait))
        nondiag_sequences[3].append(([], mw_pi2_rf))
        nondiag_sequences[3].append(([], mw_wait))
        nondiag_sequences[3].append((['mw_13'], mw_pi))
        nondiag_sequences[3].append(([], mw_wait))
        #nondiag_sequences[3].append(([], max_time - hard_pi_t - t_rot_rf - mw_pi2_rf - mw_pi)) 
        heating_time.append(max_time - hard_pi_t - t_rot_rf - mw_wait - t_rot_e_y - t_rot_e_x - mw_wait - mw_pi2_rf - mw_wait - mw_pi - mw_wait)


        sequences  = norm_sequences + diag_sequences + nondiag_sequences

        sequence = [
            (['awg_rf'],  trig_interval_rf),
            ([], trig_delay_rf),
            ([], heating_time[0] - 100),#consider (['sequence'], 100)
            (['aom'], laser),
            (['aom'], (576+512) * 1e9 /self.sampling_rf + 450 + 4 * mw_wait),
            ([], wait )
        ]
  
        for j in range(12):
            sequence.append((['awg_rf'],  trig_interval_rf))
            '''
            if j ==8:
                sequence.append((['ch2'], trig_delay_rf))
            else:
                sequence.append(([], trig_delay_rf))
            '''
            sequence.append(([], trig_delay_rf))
            sequence += sequences[j]
            sequence.append(([], heating_time[j]))
            sequence.append((['laser', 'aom'], laser))
            sequence.append((['aom' ], (576+512) * 1e9 /self.sampling_rf + 450 + 4 * mw_wait)) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9), +2000 is for 2 wait of mw between rf
            sequence.append(([], wait))
            

        sequence.append((['sequence'], 100))
        
        #self.heating_time = heating_time
        self.one_round_time = sequence_length(sequence)
        return sequence



    def VQE_wfm_rf(self,j):

        heating_time = [i * self.sampling_rf / 1e9 for i in self.heating_time]
        
        freq_awg_rf_heating = self.freq_awg_rf_heating/self.sampling_rf
        T_heating = 1/freq_awg_rf_heating

        
        rf_heating = Sin(0, freq=freq_awg_rf_heating, amp=self.amp1_rf, phase=0)


        sub_waves_rf = []

        t_2pi_rf_34 = self.t_2pi_rf_34 * self.sampling_rf / 1e9
        t_2pi_mw_24_rft = self.t_2pi_mw_24 * self.sampling_rf / 1e9
        t_2pi_mw_13_rft = self.t_2pi_mw_13 * self.sampling_rf / 1e9
        t_2pi_rf_12 = self.t_2pi_rf_12 * self.sampling_rf / 1e9
        
        t_2pi_mw_all_y = self.t_2pi_mw_all_y * self.sampling_rf / 1e9
        t_2pi_mw_all_x = self.t_2pi_mw_all_x * self.sampling_rf / 1e9
        t_rot_e_y = self.rot_e_y  * t_2pi_mw_all_y
        t_rot_e_x = self.rot_e_x  * t_2pi_mw_all_x
        t_mw_rot = t_rot_e_y + t_rot_e_x

        if self.rot_n_y > 0:
            rf_y_dir = 0
        else:
            rf_y_dir = 1

        if self.rot_n_x > 0:
            rf_x_dir = 0
        else:
            rf_x_dir = 1

        rot_n_y = abs(self.rot_n_y)
        rot_n_x = abs(self.rot_n_x)

        mw_wait_rf = self.mw_wait * self.sampling_rf / 1e9

        mw_pi = t_2pi_mw_13_rft/2
        mw_pi_2 = t_2pi_mw_24_rft/2
        mw_pi_rf = t_2pi_rf_34/2
        mw_pi_rf_12 = t_2pi_rf_12/2

        '''
        if mw_pi > mw_pi_2:
            hard_pi_t_rft = mw_pi
        else:
            hard_pi_t_rft = mw_pi_2
        '''
        hard_pi_t_rft = 0.5 * t_2pi_mw_all_x

        hard_pi_2_t_rft = 0.5 * t_2pi_mw_all_x

        freq_rf_34 = self.freq_rf_34/self.sampling_rf
        T_rf_34 = 1/freq_rf_34
        freq_rf_12 = self.freq_rf_12/self.sampling_rf
        T_rf_12 = 1/freq_rf_12
        
        phase_23 = self.phase_23
        phase_14 = self.phase_14
        
        zero_rf = Idle(1)
        idle_rf = Waveform610('IDLE_rf', [Idle(512)], sampling=self.sampling_rf)

        carry_pi13 = Idle(0)
        carry_pi13.duration = int(mw_pi) 

        
        carry_pi24 = Idle(0)
        carry_pi24.duration = int(mw_pi_2) 
        
        carry_hard_pi = Idle(0)
        carry_hard_pi.duration = int(hard_pi_t_rft) 
        
        carry_hard_pi_2 = Idle(0)
        carry_hard_pi_2.duration = int(hard_pi_2_t_rft) 
        
        carry_mw_rot = Idle(0)
        carry_mw_rot.duration = int(t_mw_rot) 

        carry_wait = Idle(0)
        carry_wait.duration = int(mw_wait_rf) # wait for the mw to finish, add to before and after mw

        

        # pulse objects


        '''
        these duration need to be changed to int numbers of rf period

        pi2x_rf_p = [Sin(t_2pi_rf_34/4, freq=freq_rf_34, amp=self.amp_rf_34, phase= 0)]
        pi2x_rf_m = [Sin(t_2pi_rf_34/4, freq=freq_rf_34, amp=self.amp_rf_34, phase= np.pi)]

        pi_34_rf_x2 = [Sin(t_2pi_rf_34, freq=freq_rf_34, amp=self.amp_rf_34, phase= 0)]
        '''
        
        duration_pi2y_rf = int(int((t_2pi_rf_34/4)/T_rf_34) * T_rf_34)
        phase_p_23 = np.pi/2. + phase_23
        pi2y_rf_p_23 = [Sin(duration_pi2y_rf, freq=freq_rf_34, amp=self.amp_rf_34, phase= phase_p_23)]
        phase_m_23 = np.pi*3/2. + phase_23
        pi2y_rf_m_23 = [Sin(duration_pi2y_rf, freq=freq_rf_34, amp=self.amp_rf_34, phase= phase_m_23)]
        phase_p_14 = np.pi/2. + phase_14
        pi2y_rf_p_14 = [Sin(duration_pi2y_rf, freq=freq_rf_34, amp=self.amp_rf_34, phase= phase_p_14)]
        phase_m_14 = np.pi*3/2. + phase_14
        pi2y_rf_m_14 = [Sin(duration_pi2y_rf, freq=freq_rf_34, amp=self.amp_rf_34, phase= phase_m_14)]

        duration_pi_34_rf = int(int((t_2pi_rf_34/2)/T_rf_34) * T_rf_34)
        pi_34_rf = [Sin(duration_pi_34_rf, freq=freq_rf_34, amp=self.amp_rf_34, phase= 0)]


        duration_pi_12_rf = int(int((t_2pi_rf_12/2)/T_rf_12) * T_rf_12)
        pi_12_rf = [Sin(duration_pi_12_rf, freq=freq_rf_12, amp=self.amp_rf_12, phase= 0)]



        
        rot_ny_duration = int(int((rot_n_y  * t_2pi_rf_34)/T_rf_34) * T_rf_34) 
        rot_nx_duration = int(int((rot_n_x  * t_2pi_rf_34)/T_rf_34) * T_rf_34) 

        phase_rot_y = np.pi/2. + rf_y_dir * np.pi
        rot_ny = Sin(rot_ny_duration, freq=freq_rf_34, amp=self.amp_rf_34, phase= phase_rot_y)
        phase_rot_x = rf_x_dir * np.pi
        rot_nx = Sin(rot_nx_duration, freq=freq_rf_34, amp=self.amp_rf_34, phase= phase_rot_x)
        

        carry_makeup1 = Idle(0)
        t_makeup1 = hard_pi_t_rft + rot_n_y * t_2pi_rf_34 + rot_n_x * t_2pi_rf_34 - carry_hard_pi.duration - rot_ny_duration - 1 - rot_nx_duration # 1 is for zero_rf in pulse_rot
        carry_makeup1.duration = int(t_makeup1) 





        pulse_rot = [rot_ny, zero_rf, rot_nx, carry_makeup1, carry_wait, carry_mw_rot, carry_wait]
        
        '''for rho23'''
        if t_rot_e_x > hard_pi_2_t_rft:
            t_rot_e_x_23 = t_rot_e_x - hard_pi_2_t_rft
        else:
            t_rot_e_x_23 = t_rot_e_x + hard_pi_2_t_rft
        t_mw_rot_23 = t_rot_e_y + t_rot_e_x_23
        carry_mw_rot_23 = Idle(0)
        carry_mw_rot_23.duration = int(t_mw_rot_23)
        
        carry_makeup2 = Idle(0)
        t_makeup2 = 2 * (mw_wait_rf - carry_wait.duration) + t_mw_rot_23 - carry_mw_rot_23.duration + t_2pi_rf_34/4 - duration_pi2y_rf
        carry_makeup2.duration = int(t_makeup2) 
 
        pulse_rot_23 = [rot_ny, zero_rf, rot_nx, carry_makeup1, carry_wait, carry_mw_rot_23, carry_wait]

        '''for rho14'''
        carry_makeup3 = Idle(0)
        t_makeup3 = 2 * (mw_wait_rf - carry_wait.duration) + t_mw_rot - carry_mw_rot.duration + t_2pi_rf_34/4 - duration_pi2y_rf
        carry_makeup3.duration = int(t_makeup3) 

        rf_heating.duration = int(int(heating_time[j]/T_heating) * T_heating)

        if j == 0:
            'N1'
            pulse_sequence_rf = [zero_rf] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_N1'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file

            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)

        elif j == 1:
            'N2'
            pulse_sequence_rf = [zero_rf] + pi_12_rf + [zero_rf] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_N2'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file

            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)
    
        elif j == 2:
            'N3'
            pulse_sequence_rf = [carry_hard_pi] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_N3'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file

            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)

            
        elif j == 3:
            'N4'
            pulse_sequence_rf = [carry_hard_pi] + pi_34_rf + [zero_rf] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_N4'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
            
            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)

            
        elif j == 4:
            'Tomo 0'
            pulse_sequence_rf = [carry_hard_pi] + pulse_rot + [carry_pi13] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_0'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
            
            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)

            
        elif j == 5:
            'Tomo pi13'
            pulse_sequence_rf = [carry_hard_pi] + pulse_rot + [zero_rf] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_pi13'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
            
            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)

            
        elif j == 6:
            'Tomo pi34'
            pulse_sequence_rf = [carry_hard_pi] + pulse_rot + [carry_pi13] + pi_34_rf + [zero_rf] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_pi34'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
            
            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)

            
        elif j == 7:
            'Tomo Pi'
            pulse_sequence_rf = [carry_hard_pi] + pulse_rot + [carry_pi24] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_Pi'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
            
            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)

            
        elif j == 8:
            'rho23 +pi/2'
            pulse_sequence_rf = [carry_hard_pi] + pulse_rot_23 + pi2y_rf_p_23 + [carry_makeup2, carry_wait, carry_pi13, carry_wait] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_rho23_p'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
            

            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)

            
        elif j == 9:
            'rho23 -pi/2'
            pulse_sequence_rf = [carry_hard_pi] + pulse_rot_23 + pi2y_rf_m_23 + [carry_makeup2, carry_wait, carry_pi13, carry_wait] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_rho23_m'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
            
            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)

            
        elif j == 10:
            'rho14 +pi/2'
            pulse_sequence_rf = [carry_hard_pi] + pulse_rot + pi2y_rf_p_14 + [carry_makeup3, carry_wait, carry_pi13, carry_wait] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_rho14_p'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
            
            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)

            
        elif j == 11:
            'rho14 -pi/2'
            pulse_sequence_rf = [carry_hard_pi] + pulse_rot + pi2y_rf_m_14 + [carry_makeup3, carry_wait, carry_pi13, carry_wait] + [rf_heating] + [zero_rf]
            name_rf = 'VQE_rho14_m'
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
            
            sub_rf = wave_rf
            sub_waves_rf.append(wave_rf)



                
        return sub_rf, sub_waves_rf





        
    def compile_waveforms(self):
        heating_time = [i * self.sampling_rf / 1e9 for i in self.heating_time]

        waves_rf = []
        idle_rf = Waveform610('IDLE_rf', [Idle(512)], sampling=self.sampling_rf)
        waves_rf.append(idle_rf)

        zero_rf = Idle(1)

        
        freq_awg_rf_heating = self.freq_awg_rf_heating/self.sampling_rf
        T_heating = 1/freq_awg_rf_heating

        
        rf_heating = Sin(0, freq=freq_awg_rf_heating, amp=self.amp1_rf, phase=0)


        main_seq_rf=Sequence('VQE_rf')
        

        'for the pulse before N1'
        rf_heating.duration = int(int((heating_time[0]-100)/T_heating) * T_heating)
        pulse_sequence_rf = [zero_rf] + [rf_heating] + [zero_rf]
        name_rf = 'VQE_nothing'
        wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
        waves_rf.append(wave_rf)
        main_seq_rf.append(wave_rf, wait=True)


        for j in range(12): 
            sub_rf, sub_waves_rf = self.VQE_wfm_rf(j)

            waves_rf += sub_waves_rf
            main_seq_rf.append(sub_rf, wait=True)

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


    get_set_items = Pulsed.get_set_items + [ 'splitting', 'y1', 'spin_state', 'rot_n_y', 'rot_n_x','rot_e_y', 'rot_e_x', 'main_dir_rf', 'amp1_rf','amp_rf_34','amp_rf_12', 'freq_rf_34', 'freq_rf_12',  't_2pi_rf_34', 't_2pi_rf_12', 't_2pi_mw_13',  't_2pi_mw_24', 'freq_awg_rf_heating', 'saveWaitTime', 'auto_save_dir', 'save_count_name', 'one_round_time', 't_2pi_mw_all_y',  't_2pi_mw_all_x', 'mw_wait', 'phase_23', 'phase_14']

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

        self.index_range = DataRange1D(self.x_data, low_setting=-0.5, high_setting=11.5, stretch_data=False)
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
                            Item('auto_save', width= -80),
                            Item('state_auto_save', width= -80, style='readonly'),
                            Item('saveWaitTime', width= -80, enabled_when='state != "monitoring"'),
                            Item('auto_save_dir', width= -240),
                            Item('one_round_time', width= -80, style='readonly'),
                            label='auto save',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        HGroup(
                            Item('reload_awg_rf'),
                            Item('wfm_button_rf', show_label=False),
                            Item('upload_progress_rf', style='readonly', format_str='%i'),
                            Item('main_dir_rf', style='readonly'),
                            Item('amp1_rf', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80, enabled_when='state != "run"'),
                            label='AWG_rf',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        HGroup(
                            Item('frequency', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                            Item('power', width=-40, enabled_when='state != "run"'),
                            Item('t_2pi_mw_13', width= -80, enabled_when='state != "run"'),
                            Item('splitting', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                            Item('frequency2', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.6f GHz' % (x*1e-9))),
                            Item('power2', width=-40, enabled_when='state != "run"'),
                            Item('t_2pi_mw_24', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80, enabled_when='state != "run"'),
                            label='SMIQ',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        HGroup(
                            Item('frequency3', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.6f GHz' % (x*1e-9))),
                            Item('power3', width=-40, enabled_when='state != "run"'),
                            Item('t_2pi_mw_all_y', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80, enabled_when='state != "run"'),
                            Item('t_2pi_mw_all_x', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80, enabled_when='state != "run"'),
                            label='SMIQ 3',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        HGroup(
                            Item('rot_n_y', width= -80, enabled_when='state != "run"'),
                            Item('rot_n_x', width= -80, enabled_when='state != "run"'),
                            Item('rot_e_y', width= -80, enabled_when='state != "run"'),
                            Item('rot_e_x', width= -80, enabled_when='state != "run"'),
                            Item('mw_wait', width= -80, enabled_when='state != "run"'),
                            Item('phase_23', width= -80, enabled_when='state != "run"'),
                            Item('phase_14', width= -80, enabled_when='state != "run"'),
                            label='VQE parameters',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item('freq_rf_34', width= -80, enabled_when='state != "run"'),
                        Item('amp_rf_34', width= -80, enabled_when='state != "run"'),
                        Item('t_2pi_rf_34', width= -80, enabled_when='state != "run"'),
                        Item('freq_awg_rf_heating', width= -80, enabled_when='state != "run"'),
                        label='rf',
                        show_border=True,
                    ),
                    HGroup(
                        Item('freq_rf_12', width= -80, enabled_when='state != "run"'),
                        Item('amp_rf_12', width= -80, enabled_when='state != "run"'),
                        Item('t_2pi_rf_12', width= -80, enabled_when='state != "run"'),
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
        title='VQE_2p',
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
    
