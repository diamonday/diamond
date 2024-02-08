import numpy as np
import logging
import time

from traits.api import Trait, Range, Int, Float, Complex, Bool, Tuple, Array, Str, Instance, Property, Enum, on_trait_change, Button, DelegatesTo
from traitsui.api import View, Item, Tabbed, HGroup, VGroup, VSplit, EnumEditor, TextEditor
from traitsui.menu import Action, Menu, MenuBar
from traitsui.file_dialog import save_file

from chaco.api import VPlotContainer, HPlotContainer, OverlayPlotContainer, Plot, CMapImagePlot, ArrayPlotData, ArrayDataSource, LinePlot, DataRange1D, jet, reverse, create_line_plot, create_polar_plot
from chaco.api import Spectral, LinearMapper, DataLabel, PlotLabel, ColorBar      
from chaco.tools.api import RangeSelection, RangeSelectionOverlay
from chaco.tools.cursor_tool import CursorTool2D, CursorTool, BaseCursorTool
from enable.api import Component, ComponentEditor

from hardware.api import PulseGenerator, TimeTagger, Microwave#, MicrowaveD, MicrowaveE, RFSource
from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin, GetSetItemsHandler
from measurements.pulsed import Pulsed as measurePulsed
from analysis.fitting import find_edge, run_sum

from analysis import fitting
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
    edge = find_edge(profile)
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
    
class SNSC2dHandler(GetSetItemsHandler):
    def saveColorPlot(self, info):
        filename = save_file(title='Save Plot')
        if filename is '':
            return
        else:
            if filename.find('.png') == -1:
                filename = filename + '.png'
            info.object.save_color_plot(filename)
    
    
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
            info.object.save_all_figure(filename)
            info.object.save(filename)

class SNSCXYn2d(measurePulsed, ManagedJob, GetSetItemsMixin):
    '''
    +======================================================================================================+
    |           SNSCXYn_2d: symmetric and non-symmetric 2d correlation spectroscopies                      |
    |           Changed from SNSCXYn. Mainly changed the sequence and UI                                   |
    |           Defines two sequences                                                                      |
    |           Symmetric correlation spectrometry:                                                        |
    |           |I|~~~~~~~~1~~~~~~~~|*|_|**|_|^|~~~~~~~~~2~~~~~~~|*|_|**|_|^|___|R|                        |
    |                                                                                                      |
    |           Non-symmetric correlation spectrometry:(Not started yet)                                   |
    |                                                                                                      |
    |           |I|~~~~~~~~1~~~~~~~~|*|_|**|_| |~~~~~~~~~2~~~~~~~|*|_|**|_|^|___|R|                        |
    |                                                                                                      |
    |           |I|: Initialize                                                                            |
    |           |R|: Readout                                                                               |
    |           |*|: \pi/2 pulse in y direction                                                            |
    |           |**|: \pi pulse in x direction                                                             |
    |           |^|: \pi/2 pulse in x direction                                                            |
    |           | |: \pi/2 pulse in z direction                                                            |
    |                _: tau_DD                                                                             |
    |           ~~~1~~~: tau_f_1                                                                           |
    |           ~~~2~~~: tau_f_2                                                                           |
    |           ___: waiting time                                                                          |
    +======================================================================================================+
    |           Enhancement pulse implement before readout                                                 |
    |           |rf1|==|rf2|                                                                               |
    |                                                                                                      |
    |           |rf1|: \pi pulse from |-1,+1> to |-1,0>                                                    |
    |           |rf2|: \pi pulse from |-1,0> to |-1,-1>                                                    |
    |            ==: waitp                                                                                 |
    |                                                                                                      |
    |                                                                                                      |
    |                                                                                                      |
    +======================================================================================================+
    '''     

    #measurement parameters
    laserTime = Range(low=1.5, high=1.0e+9, value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=0., high=100000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
      
    frequency = Range(low=1., high=20e9, value=1.55e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    power = Range(low= -100., high=16., value= -10.0, desc='power of microwave', label='power[dBm]', mode='text', auto_set=False, enter_set=True)

    tau_DD=Range(low=0.0, high=1.0e+6, value=2.0e+3, desc='time interval of a DD block',label='tau_DD[ns]', mode='text',auto_set=False, enter_set=True)
    
    t_2pi_x = Range(low=1., high=100000., value=200., desc='rabi period 2pi pulse length (x)', label='2pi x [ns]', mode='text', auto_set=False, enter_set=True)
    t_rabi = Range(low=1., high=100000., value=200., desc='rabi period 2pi pulse length (x)', label='initial t [ns]', mode='text', auto_set=False, enter_set=True)
    t_2pi_y = Range(low=1., high=100000., value=200., desc='rabi period 2pi pulse length (y, 90 degree)', label='2pi y[ns]', mode='text', auto_set=False, enter_set=True)

    tau_f_start = Range(low=0.0, high=1.0e+8, value=1.5, desc='starting time of free evolution[ns]', label='tau_f start[ns]', mode='text', auto_set=False, enter_set=True)
    tau_f_end = Range(low=1.5, high=1.0e+9, value=300.0e3, desc='ending time of free evolution[ns]', label='tau_f end[ns]', mode='text', auto_set=False, enter_set=True)
    delta_tau_f = Range(low=1.5, high=1.0e+6, value=300.0, desc='delta time of free evolution[ns]', label='delta tau_f[ns]', mode='text', auto_set=False, enter_set=True)
    n_pi = Range(low=0, high=300, value=2, desc='number of XY', label='XY-n', mode='text', auto_set=False, enter_set=True)
    n_ref = Range(low=0, high=300, value=0, desc='number of ref', label='n ref', mode='text', auto_set=False, enter_set=True)

    #doSorSNS = Enum("SandNS", "S", desc='perform S or S&NS measurement', label='do S/NS')

    #initialize nuclear spin by SSR
    polarizeNspinSSR = Bool(False, desc='polarize nuclear spin by SSR', label='P nspin SSR')
    repetition_SSR = Range(low=1, high=1000, value=100, desc='number of repetition of a block', label='repetition', mode='text', auto_set=False, enter_set=True)
    laserTime_SSR = Range(low=1.5, high=100000., value=300., desc='laser duration[ns]', label='laser[ns]', mode='text', auto_set=False, enter_set=True)
    wait_SSR = Range(low=1.5, high=100000., value=300., desc='waiting time after laser[ns]', label='wait[ns]', mode='text', auto_set=False, enter_set=True)
    n_pi_SSR = Range(low=1, high=300, value=16, desc='number of pi pulses ', label='n pi', mode='text', auto_set=False, enter_set=True)
    tau_DD_SSR = Range(low=1.5, high=1.0e+6, value=2.0e+3, desc='time interval of the DD block',label='tau_DD[ns]', mode='text',auto_set=False, enter_set=True)
    t_pi_SSR = Range(low=1.5, high=100000., value=100.0, desc='pi pulse length', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    #initialize nuclear spin by PuPol
    polarizeNspinPuPol = Bool(False, desc='polarize nuclear spin by PuPol', label='P nspin PuPol')
    repetition_PuPol = Range(low=1, high=3000, value=50, desc='number of repetition of a block', label='repetition PP', mode='text', auto_set=False, enter_set=True)
    repetition_P = Range(low=1, high=3000, value=20, desc='number of repetition of a SNS unit', label='repetition DD', mode='text', auto_set=False, enter_set=True)
    laserTime_PuPol = Range(low=1.5, high=100000., value=300., desc='laser duration[ns]', label='laser[ns]', mode='text', auto_set=False, enter_set=True)
    wait_PuPol = Range(low=1.5, high=100000., value=300., desc='waiting time after laser[ns]', label='wait[ns]', mode='text', auto_set=False, enter_set=True)
    tau_PuPol = Range(low=1.5, high=1.0e+6, value=2.0e+3, desc='time interval of the SNS block',label='tau_PP[ns]', mode='text',auto_set=False, enter_set=True)

    espin0or1 = Enum("no rf", "0", "1", desc='with espin 0 or 1 when rf, or not apply rf', label='with 0/1')
    reinitespin = Bool(True, desc='re-initialize e spin after rf', label='re-init epsin')
    t_rf2 = Range(low=1.5, high=8000000., value=80., desc='rf2 pulse length', label='rf2 [ns]', mode='text', auto_set=False, enter_set=True) #for manipulating C13
    #for enhancement readout
    enhancementreadout = Bool(False, desc='enhancement readout', label='enhance')
    waitp = Range(low=0., high=100000., value=1000., desc='waitp [ns]', label='waitp [ns]', mode='text', auto_set=False, enter_set=True)
    waith = Range(low=0., high=2000000., value=0., desc='waith [ns]', label='wait for heating [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi_rf1 = Range(low=1.5, high=100000., value=80., desc='rf1 pi pulse length', label='rf1 pi[ns]', mode='text', auto_set=False, enter_set=True) #-1,+1 to -1,0
    t_pi_rf2 = Range(low=1.5, high=100000., value=80., desc='rf2 pi pulse length', label='rf2 pi[ns]', mode='text', auto_set=False, enter_set=True) #-1,0 to -1,-1
    rf1Frequency = Range(low=1., high=20e9, value=5.0e6, desc='radio frequency', label='rf1 frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    rf1Power = Range(low= -100., high=16., value= -10, desc='power of radio frequency', label='rf1 power[dBm]', mode='text', auto_set=False, enter_set=True)
    rf2Frequency = Range(low=1., high=20e9, value=5.0e6, desc='radio frequency', label='rf2 frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    rf2Power = Range(low= -100., high=16., value= -10, desc='power of radio frequency', label='rf2 power[dBm]', mode='text', auto_set=False, enter_set=True)
    total_time = Str("0.0", label="sequence time")

    #frequency domain information
    resolution = Str('0.0 Hz', label='resolution')
    freqRange = Str('spectrum range: 0.0 Hz', label='spectrum range')
    numDataPoint = Str('0', label='point number')
    # n_ref = Range(low=0, high=100, value=0, desc='number of reference pulses', label='n ref', mode='text', auto_set=False, enter_set=True)
    # performNS = Bool(False, desc="perform anit-Symmetric", label="perform anit-Symmetric")

    tau_f_seq = Array(value=np.array((0., 1.)))
    
    #result data
    pulse = Array(value=np.array((0., 0.)))
    flank = Float(value=0.0)
    spin_state = Array(value=np.array((0., 0.)))
    spin_state_error = Array(value=np.array((0., 0.)))
    #spin_state2d = Array(value=np.array((0., 0.))) # did not include the n_ref spin state
    #spin_state2d_error = Array(value=np.array((0., 0.)))
    integration_width = Range(low=10., high=1000., value=200., desc='time window for pulse analysis [ns]', label='integr. width [ns]', mode='text', auto_set=False, enter_set=True)
    position_signal = Range(low= -100., high=1000., value=0., desc='position of signal window relative to edge [ns]', label='pos. signal [ns]', mode='text', auto_set=False, enter_set=True)
    position_normalize = Range(low=0., high=10000., value=2200., desc='position of normalization window relative to edge [ns]', label='pos. norm. [ns]', mode='text', auto_set=False, enter_set=True)
    
    
    spectrum_FT_freq = Array(value=np.array((0., 1.)))
    spectrum_FT_S = Array(value=np.array((0., 1.)))
    #spectrum_FT_NS = Array(value=np.array((0., 1.)))
    brightRef = Float(value=0.0)
    darkRef = Float(value=0.0)

    thresh_high = Trait('auto', Str('auto'), Float(10000.), desc='High Limit of image plot', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float))
    thresh_low = Trait('auto', Str('auto'), Float(0.), desc='Low Limit of image plot', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float))
    colorbar = Instance(ColorBar)

    #fit data
    #perform_fit = Bool(False, label='perform fit')
    #para_S_fit = Array(value=np.array((0., 0., 0.)))
    #para_NS_fit = Array(value=np.array((0., 0., 0.)))
    # para_dev_S_fit = Tuple(0,0,0)
    # para_dev_NS_fit = Tuple(0,0,0)
    #fitData_display = Str("None", label="fit data") #display fit data

    #phase_shift = Complex(0+0j) # Phase shift due to initial t =/= 0
    #peak_S = Complex(0+0j) #selected peak to plot its complex value
    #peak_NS = Complex(0+0j)

    #polarization_selected = Float(0.0, label="polarization")
    #plotting
    show_raw_data = Bool(False, label='show raw data as matrix plot')
    matrix_plot_data = Instance(ArrayPlotData) #raw data of spin state
    matrix_plot = Instance(Plot, editor=ComponentEditor())

    figureContainer = Instance(VPlotContainer) #a box contains all the colormap plot, scatter plots and colorbar
    
    scatterPlot_t1_start_Data = Instance(ArrayPlotData) # plot data for symmetric signal of t2 at fixed t1, t1 is the start of tau_f sequence
    scatterPlot_t1_end_Data = Instance(ArrayPlotData) # plot data for symmetric signal of t2 at fixed t1, t1 is the end of tau_f sequence
    scatterPlot_t2_start_Data = Instance(ArrayPlotData) # plot data for symmetric signal of t1 at fixed t2, t2 is the start of tau_f sequence
    scatterPlot_t2_end_Data = Instance(ArrayPlotData) # plot data for symmetric signal of t1 at fixed t2, t2 is the end of tau_f sequence
    scatterPlot_t1_start = Instance(Plot) #scatter plot for symmetric signal of t2 at fixed t1, t1 is the start of tau_f sequence
    scatterPlot_t1_end = Instance(Plot) #scatter plot for symmetric signal of t2 at fixed t1, t1 is the end of tau_f sequence
    scatterPlot_t2_start = Instance(Plot) #scatter plot for symmetric signal of t1 at fixed t2, t2 is the start of tau_f sequence
    scatterPlot_t2_end = Instance(Plot) #scatter plot for symmetric signal of t1 at fixed t2, t2 is the end of tau_f sequence

    #correlation_spectrum_all_data  = Instance(ArrayPlotData) # plot data for ploarization
    #correlation_spectrum_all = Instance(Component) # plot for polarization , i.e. -Im(NS)/Re(S)

    correlation_spectrum_Sdata = Instance(ArrayPlotData) # plot data for Fourier transform of the symmetric
    #correlation_spectrum_NSdata = Instance(ArrayPlotData) # plot data for Fourier transform of the non-symmetric
    correlation_spectrum_S = Instance(Plot) # plot for Fourier transform of the symmetric
    #correlation_spectrum_NS = Instance(Plot) # plot for Fourier transform of the non-symmetric
    '''
    polarPlot = Instance(Plot)
    #plot tools
    rangeSelector = Instance(RangeSelection)
    selectedRange_1 = Array(value=np.array((0, -1))) #selected range of frequency in terms of index
    selectedRange_2 = Array(value=np.array((0, -1))) #selected range of frequency in terms of index
    # range_display = Str("None", label="selected range") #display selected range
    cursor_S_1 = Instance(BaseCursorTool)
    cursor_S_2 = Instance(BaseCursorTool)
    cursor_NS_1 = Instance(BaseCursorTool)
    cursor_NS_2 = Instance(BaseCursorTool)
    cursorPosition_1 = Array(value=np.array((0.5, 0)))
    cursorPosition_2 = Array(value=np.array((0.5, 0)))
    cursorFreq_display_1 = Float(0.0, label="cursor freq(MHz)")
    cursorFreq_display_2 = Float(0.0, label="cursor freq(MHz)")
    '''
    def __init__(self):
        super(SNSCXYn2d, self).__init__()


        self.get_freqInformation()

        #create different plots  
        self._create_matrix_plot()
        self._create_pulse_plot()
        #self._creat_polarPlot()
        #self._create_correlation_spectrum_all()
        #self._create_scatterPlot_S()
        #self._create_scatterPlot_NS()
        self._create_scatterPlot_t1_start()
        self._create_scatterPlot_t1_end()
        self._create_scatterPlot_t2_start()
        self._create_scatterPlot_t2_end()
        self._create_correlation_spectrum_S()
        #self._create_correlation_spectrum_NS()
        self._create_figureContainer()

        self.sync_trait('thresh_high', self.correlation_spectrum_S.plots['image'][0].value_range, 'high_setting')
        self.sync_trait('thresh_low', self.correlation_spectrum_S.plots['image'][0].value_range, 'low_setting')
        self.on_trait_change(self.correlation_spectrum_S.plots['image'][0].request_redraw, 'thresh_high', dispatch='ui')
        self.on_trait_change(self.correlation_spectrum_S.plots['image'][0].request_redraw, 'thresh_low', dispatch='ui')

        '''
        #tool
        self.sync_trait('cursorPosition_1', self.cursor_S_1, 'current_position')
        self.sync_trait('cursorPosition_2', self.cursor_S_2, 'current_position')
        self.sync_trait('cursorPosition_1', self.cursor_NS_1, 'current_position')
        self.sync_trait('cursorPosition_2', self.cursor_NS_2, 'current_position')
        '''
    #==========================================================|
    #               check parameters, set devices              |
    #==========================================================|
    def start_up(self):
        PulseGenerator().Night()
        Microwave().setOutput(self.power, self.frequency)
        #if self.enhancementreadout or self.polarizeNspinPuPol:
           # MicrowaveD().setOutput(self.rf2Power, self.rf2Frequency) #for enhancement #3 mwD
        #    RFSource().setOutput(self.rf1Power, self.rf1Frequency) #for enhancement
        #else:
           # MicrowaveD().setOutput(None, self.rf2Frequency) #for enhancement
        #   RFSource().setOutput(-22, self.rf1Frequency) #for enhancement

    def shut_down(self):
        PulseGenerator().Light()
        Microwave().setOutput(None, self.frequency)
        #MicrowaveD().setOutput(None, self.rf2Frequency) 
        # RFSource().setOutput(-22, self.rf1Frequency)
        
    def generate_sequence(self):
        laserTime = self.laserTime
        wait = self.wait
        waith = self.waith

        #for enhancement
        waitp = self.waitp      
        t_pi_rf1 = self.t_pi_rf1
        t_pi_rf2 = self.t_pi_rf2
        t_rabi = self.t_rabi

        t_2pi_x = self.t_2pi_x
        t_2pi_y = self.t_2pi_y
        t_pi2_x = t_2pi_x/4.0
        t_pi2_y = t_2pi_y/4.0
        t_pi_x = t_2pi_x/2.0 
        t_pi_y = t_2pi_y/2.0

        n_pi = self.n_pi
        N_ref = self.n_ref

        tau_f_seq = np.arange(self.tau_f_start, self.tau_f_end, self.delta_tau_f)
        self.tau_f_seq = tau_f_seq
        tau_f_1_seq = tau_f_seq
        tau_f_2_seq = tau_f_seq

        #initialization part
        if self.polarizeNspinPuPol and (self.polarizeNspinSSR == False):
            espin0or1 = self.espin0or1
            reinitespin = self.reinitespin
            t_rf2 = self.t_rf2
            laserTime_PuPol = self.laserTime_PuPol
            wait_PuPol = self.wait_PuPol
            tau_PuPol = self.tau_PuPol
            repetition_PuPol = self.repetition_PuPol
            repetition_P = self.repetition_P

            tPP = tau_PuPol-(t_pi2_y+t_pi2_x)*2.0-(t_pi_x+t_pi_y)*1.0
            if tPP < 0:
                tPP = 0
            else:
                tPP = tPP
            DDseq = [ (['mw_x'], t_pi2_x), ([], tPP/4.0), (['mw_y'], t_pi_y), ([], tPP/4.0), (['mw_x'], t_pi2_x),
                        (['mw_y'], t_pi2_y), ([], tPP/4.0), (['mw_x'], t_pi_x), ([], tPP/4.0), (['mw_y'], t_pi2_y)]*repetition_P 
            initial = ( [(['aom'], laserTime_PuPol), ([], wait_PuPol), (['mw_x'], t_pi_x/2.0), ([], wait_PuPol)] + DDseq) * repetition_PuPol
            if t_rabi == "0":
                initial1 = ( [(['aom'], laserTime_PuPol), ([], wait_PuPol)] + DDseq) * repetition_PuPol
            else:
                initial1 = ( [(['aom'], laserTime_PuPol), ([], wait_PuPol), (['mw_x'], t_rabi), ([], wait_PuPol)] + DDseq) * repetition_PuPol
               # initial1 = ( [(['aom'], laserTime_PuPol), ([], wait_PuPol)] + DDseq) * repetition_PuPol
            initial += [(['aom'], laserTime), ([], wait)]
            initial1 += [(['aom'], laserTime), ([], wait)]

            
            #apply rf to rotate the nuclear spin
            if espin0or1 == "0":
                initial1 += [(['rf'], t_rf2/2.0)]
                initial1 += [(['rf'], t_rf2/2.0)]
                initial += [([], t_rf2/2.0)]
                initial += [([], t_rf2/2.0)]
                if reinitespin:
                    initial += [(['aom'], laserTime), ([], wait)]
                    initial1 += [(['aom'], laserTime), ([], wait)]
                else:
                    initial += [([], wait_PuPol)]
                    initial1 += [([], wait_PuPol)]
            elif espin0or1 == "1":
                # initial += [(['mw_x'], t_pi_x), (['rf'], t_rf2/2.0)]
                # initial += [(['mw_x'], t_pi_x), (['rf'], t_rf2/2.0)]
                initial1 += [([], t_rf2/2.0)]
                initial1 += [([], t_rf2/2.0)]
                initial += [([], t_rf2/2.0)]
                initial += [([], t_rf2/2.0)]
                if reinitespin:
                    initial += [(['aom'], laserTime), ([], wait)]
                    initial1 += [(['aom'], laserTime), ([], wait)]
                else:
                    initial += [([], wait_PuPol)]
                    initial1 += [([], wait_PuPol)]
            elif espin0or1 == "no rf":
                print("OK_pp____________")
                pass
            else:
                pass
            

        # elif self.polarizeNspinSSR and (self.polarizeNspinPuPol == False):
        #     t_pi_SSR = self.t_pi_SSR
        #     if self.tau_DD_SSR>=t_pi_x:
        #         interval_DD_SSR = self.tau_DD_SSR - t_pi_SSR
        #     else:
        #         interval_DD_SSR = 0.0
        #     repetition_SSR = self.repetition_SSR
        #     laserTime_SSR = self.laserTime
        #     wait_SSR = self.wait_SSR
        #     n_pi_SSR = self.n_pi_SSR
        #     tau_DD_SSR = self.tau_DD_SSR
        #     t_pi_SSR = self.t_pi_SSR
        #     initial = [(['aom'], laserTime_SSR), ([], wait_SSR), ([], tau_DD_SSR/2.0), (['mw'], t_pi_SSR), ([], tau_DD_SSR/2.0)]*repetition_SSR
        #     initial += [(['aom'], laserTime), ([], wait)  ]
        else:
            initial = []
            initial1 = []
        
        '''
        #enhancement readout
        if self.enhancementreadout:
            waitp = self.waitp      
            t_pi_rf1 = self.t_pi_rf1
            t_pi_rf2 = self.t_pi_rf2        
            re = [(['rf'], t_pi_rf1), ([], waitp), (['rf'], t_pi_rf2)]
        else:
            re =[]
        '''
        re =[]

        sequence = []
        
        tDD = self.tau_DD - (t_pi_y + t_pi_x)/4.0
        if tDD < 0:
            tDD = 0

        Unit_X = [([], tDD), (['mw_x'], t_pi_x), ([], tDD)]
        Unit_Y = [([], tDD), (['mw_y'], t_pi_y), ([], tDD)]
        #XY8 = (Unit_X + Unit_Y + Unit_X + Unit_Y)*int(n_pi // 8) + (Unit_Y + Unit_X + Unit_Y + Unit_X)*int(n_pi // 8)
        XY8 = Unit_X
        
        for tau_f_1 in tau_f_1_seq: #symmetric correlation
            tf_1 = tau_f_1 #- t_pi_y
            if tf_1 < 0:
                tf_1 = 0
            for tau_f_2 in tau_f_2_seq:               
                tf_2 = tau_f_2 #- t_pi_y
                if tf_2 < 0:
                    tf_2 = 0
                
                sequence += initial1 # 1/2
                
                sequence += [([], tf_1)]
                '''
                sequence += [([], tf_1/2.0)]
                sequence += [(['mw_y'], t_pi_y)]
                sequence += [([], tf_1/2.0)]
                '''
                sequence += [(['mw_x'], t_pi2_x)]
                sequence += XY8
                sequence += [(['mw_y'], t_pi2_y * 5.0)] # 2/2 5

                sequence += [([], tf_2)]
                '''
                sequence += [([], tf_2/2.0)]
                sequence += [(['mw_y'], t_pi_y)]
                sequence += [([], tf_2/2.0)]
                '''
                sequence += [(['mw_x'], t_pi2_x)]
                sequence += XY8
                sequence += [(['mw_y'], t_pi2_y)]
                #enhancement
                sequence += re
                sequence += [(['laser', 'aom'], laserTime), ([], wait)]
                if waith > 0:
                    sequence +=[(['aom'], waith)]
                    sequence += [(['aom'], laserTime), ([], wait)]
        '''
        if self.doSorSNS == "SandNS":
            for tau_f_1 in tau_f_1_seq: #non-symmetric correlation
                tf_1 = tau_f_1 - t_pi_y
                if tf_1 < 0:
                    tf_1 = 0
                for tau_f_2 in tau_f_2_seq:               
                    tf_2 = tau_f_2 - t_pi_y
                    if tf_2 < 0:
                        tf_2 = 0
                
                    sequence += initial1

                    sequence += [([], tf_1/2.0)]
                    sequence += [(['mw_y'], t_pi_y)]
                    sequence += [([], tf_1/2.0)]

                    sequence += [(['mw_x'], t_pi2_x)]
                    sequence += XY8
                    sequence += [(['mw_y'], t_pi2_y * 4.0)] # 4.0
                    sequence += [([], t_pi2_y )]

                    sequence += [([], tf_2/2.0)]
                    sequence += [(['mw_y'], t_pi_y)]
                    sequence += [([], tf_2/2.0) ]

                    sequence += [(['mw_x'], t_pi2_x)]
                    sequence += XY8
                    sequence += [(['mw_y'], t_pi2_y)]

                    #enhancement
                    sequence += re
                    sequence += [ (['laser', 'aom'], laserTime), ([], wait)  ]  
                    if waith > 0:
                        sequence += [ (['aom'], waith) ]
                        sequence += [ (['aom'], laserTime), ([], wait)  ]
        '''
        sequence += N_ref * (re + [ (['laser', 'aom'], laserTime), ([], wait)]) #bright
        sequence += N_ref * ([ (['mw_x'], t_pi_x)] + re + [(['laser', 'aom'], laserTime), ([], wait)]) #dark
        # sequence += n_ref * [ (['laser', 'aom'], laser), ([], wait)  ]
        # sequence += n_ref * [ (['mw_x'], t_pi_y), (['laser', 'aom'], laser), ([], wait)  ]
        sequence += [ (['sequence'], 100)  ]
        return sequence

    @on_trait_change('enhancementreadout')
    def calculate_total_time(self):
        s = self.generate_sequence()
        g = 0
        for i in range(len(s)):
            g += s[i][1]
        #record total time of sequence 
        if g<10**3:
            self.total_time = "{} ns".format(g)  
        elif g>10**3 and g<=10**6:
            self.total_time = "{} us".format(g/10**3)
        elif g>10**6 and g<=10**9:
            self.total_time = "{} ms".format(g/10**6)
        elif g>10**9:
            self.total_time = "{} s".format(g/10**9)


    @on_trait_change('tau_f_end, tau_f_start, delta_tau_f')
    def get_freqInformation(self):
        tau_f_seq = np.arange(self.tau_f_start, self.tau_f_end, self.delta_tau_f)
        self.tau_f_seq = tau_f_seq

        resolutioni = 1.0/(self.tau_f_end-self.tau_f_start)*10**9
        freqRangei = 0.5/self.delta_tau_f*10**9
        numDataPointi = int(2*freqRangei/resolutioni+0.5)

        if resolutioni<10**3:
            self.resolution = "{} Hz".format(round(resolutioni,1))   
        elif resolutioni>10**3 and resolutioni<=10**6:
            self.resolution = "{} kHz".format(round(resolutioni/10**3,1)) 
        elif resolutioni>10**6 and resolutioni<=10**9:
            self.resolution = "{} MHz".format(round(resolutioni/10**6,1))
        elif resolutioni>10**9:
            self.resolution = "{} GHz".format(round(resolutioni/10**9,1))

        if freqRangei<10**3:
            self.freqRange = "{} Hz".format(round(freqRangei,1)) 
        elif freqRangei>10**3 and freqRangei<=10**6:
            self.freqRange = "{} kHz".format(round(freqRangei/10**3,1))
        elif freqRangei>10**6 and freqRangei<=10**9:
            self.freqRange = "{} MHz".format(round(freqRangei/10**6,1))
        elif freqRangei>10**9:
            self.freqRange = "{} GHz".format(round(freqRangei/10**9,1))  
        self.numDataPoint = "{}".format(numDataPointi)
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
        y =  np.nan_to_num(y) #turn all NaN into 0 

        self.spin_state = y
        self.spin_state_error = y ** 0.5
        self.pulse = profile
        self.flank = self.time_bins[flank]

        # need to change back to 1D array
    @on_trait_change('spin_state')
    def update_FTspectrum(self):
        #get spin state
        if self.n_ref != 0:
            ss = self.spin_state[:-2*self.n_ref]
        else:
            ss = self.spin_state

        n_tau_f = self.tau_f_seq.size # get tau_f number

        #do FFT to symmetric, nonsymmetric parts
        y = ss
        y = y.reshape((n_tau_f,n_tau_f))

        y2 = np.zeros(( n_tau_f,2*n_tau_f - 1))

        for (a,b), element in np.ndenumerate(y):
            y2[a, a+b] = element

        for (a,b), element in np.ndenumerate(y):
            a1 = a
            b1 = a+b 
            b2 = a1
            a2 = b1
            if (( a2 >= 0 and a2 <= (n_tau_f - 1) ) and ( b2 >= 0 and b2 <= (2*n_tau_f - 2) ) ) and (y2[a2,b2] == 0) :
                y2[a2,b2] = element
        
        y3 = y2[:,:n_tau_f]

        y3 = y3-y3.mean()
        sp_S = np.fft.fft2(y3) #2dimensional fft
        #sp_NS = sp_S*0.0

        #scale the values of strength, frequency
        #scaleUnit = max(list(abs(sp_S.reshape(n_tau_f ** 2))) + list(abs(sp_NS.reshape(n_tau_f ** 2))))
        scaleUnit = max(list(abs(sp_S.reshape(n_tau_f ** 2))) )
        sp_S = sp_S/scaleUnit
        #sp_NS = sp_NS/scaleUnit
        samplingSpace = self.delta_tau_f
        freq_ft = np.fft.fftfreq(n_tau_f, samplingSpace)
        freq_ft = freq_ft*10**3 #in MHz
        # freq = np.fft.fftfreq(len(y), 1)
        # freq = freq/self.delta_tau_f*10**3 #in MHz
        # freq_ft = freq

        self.spectrum_FT_freq = np.fft.fftshift(freq_ft)
        self.spectrum_FT_S = np.fft.fftshift(sp_S)
        #self.spectrum_FT_NS = np.fft.fftshift(sp_NS)
    '''
    @on_trait_change('cursorPosition, spin_state')
    def update_selectedPeak_FTvalue(self):
        portion_1 = self.cursorPosition_1[0] #value from 0 to 1
        portion_2 = self.cursorPosition_2[0] #value from 0 to 1
        indexbound_1 = self.selectedRange_1
        indexbound_2 = self.selectedRange_2
        i0_1, i1_1 = indexbound_1[0], indexbound_1[1]
        i0_2, i1_2 = indexbound_2[0], indexbound_2[1]
        length_1 = (i1_1-i0_1)
        length_2 = (i1_2-i0_2)
        # peakIndex = int(portion*length+i0)
        # weighting = (portion*length)%1
        # self.peak_S = self.spectrum_FT_S[peakIndex]*weighting + self.spectrum_FT_S[peakIndex+1]*(1-weighting) #interpolate
        # self.peak_NS = self.spectrum_FT_NS[peakIndex]*weighting + self.spectrum_FT_S[peakIndex+1]*(1-weighting) #interpolate
        portion_1 = portion_1
        portion_2 = portion_2
        peakIndex_1 = int(round(portion_1*length_1+i0_1))
        peakIndex_2 = int(round(portion_2*length_2+i0_2))
        self.phase_shift = np.exp(-1.j*(self.spectrum_FT_freq[peakIndex_1] + self.spectrum_FT_freq[peakIndex_2])*self.tau_f_start*1e-3)
        self.peak_S = self.spectrum_FT_S[peakIndex_1,peakIndex_2]
        self.peak_NS = self.spectrum_FT_NS[peakIndex_1,peakIndex_2]

        self.cursorFreq_display_1 = self.spectrum_FT_freq[peakIndex_1]
        self.cursorFreq_display_2 = self.spectrum_FT_freq[peakIndex_2]
        self.polarization_selected = np.true_divide(abs(self.peak_NS),abs(self.peak_S))
    '''
    #==========================================================|
    #                    perform fitting                       |
    #==========================================================|
    '''
    @on_trait_change('spin_state, selectedRange, perform_fit')
    def update_fitData(self):
        index0 = self.selectedRange[0]
        index1 = self.selectedRange[1]
        if (index1-index0)>10 and self.perform_fit:
            x = self.spectrum_FT_freq[index0:index1]
            try:
                y_S = abs(self.spectrum_FT_S[index0:index1])
                para_S = fitting.fit_multiple_lorentzians(x,y_S,number_of_lorentzians='auto',threshold=0.5)
            except:
                para_S = [0,0,0]
            try:
                y_NS  = abs(self.spectrum_FT_NS[index0:index1])
                para_NS = fitting.fit_multiple_lorentzians(x,y_NS,number_of_lorentzians='auto',threshold=0.5)
            except:
                para_NS = [0,0,0]
            #display the fit data
            fdd = ""
            peakNum_S = len(self.para_S_fit)/3
            peakNum_NS = len(self.para_NS_fit)/3
            fdd += "symmetric\n"
            for i in range(peakNum_S):
                fdd += "peak{}:\nfreq(MHz) {} \nwidth(MHz) {} \n".format(i+1, self.para_S_fit[3*i+1], self.para_S_fit[3*i+2])
            fdd += "non-symmetric\n"
            for i in range(peakNum_NS):
                fdd += "peak{}:\nfreq(MHz) {} \nwidth(MHz) {} \n".format(i+1, self.para_NS_fit[3*i+1], self.para_NS_fit[3*i+2])

            self.para_S_fit = para_S
            self.para_NS_fit = para_NS
            self.fitData_display = fdd
        else:
            self.para_S_fit = [0,0,0]
            self.para_NS_fit = [0,0,0]
            self.fitData_display = "None"
    '''
    #==========================================================|
    #            create all the plots and container            |
    #==========================================================|
    def _create_matrix_plot(self):
        matrix_plot_data = ArrayPlotData(image=np.zeros((self.n_laser, self.n_bins)))
        matrix_plot = Plot(matrix_plot_data, height=300, resizable='hv', padding=8, padding_left=64, padding_bottom=32)
        matrix_plot.img_plot("image",
                              xbounds=(self.time_bins[0], self.time_bins[-1]),
                              ybounds=(0, self.n_laser),
                              colormap=Spectral,
                             )[0]
        matrix_plot.index_axis.title = 'time [ns]'
        matrix_plot.value_axis.title = 'laser pulse'
        self.matrix_plot_data = matrix_plot_data
        self.matrix_plot = matrix_plot
        
    def _create_pulse_plot(self):
        self.pulse_plot_data = ArrayPlotData(x=self.time_bins, y=self.pulse)
        plot = Plot(self.pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(('x', 'y'), style='line', color='blue', name='data')
        edge_marker = LinePlot(index=ArrayDataSource(np.array((0, 0))),
                               value=ArrayDataSource(np.array((0, 1e9))),
                               color='red',
                               index_mapper=LinearMapper(range=plot.index_range),
                               value_mapper=LinearMapper(range=plot.value_range),
                               name='marker')
        plot.add(edge_marker)
        plot.index_axis.title = 'time [ns]'
        plot.value_axis.title = 'intensity'
        self.pulse_plot = plot
    '''
    def _create_correlation_spectrum_all(self): 
        x = self.tau_f_seq
        y = x*0.0
        plotdata = ArrayPlotData(freq=x, sp_S=y, sp_NS=y)
        self.correlation_spectrum_all_data = plotdata

        plot = Plot(self.correlation_spectrum_all_data, height=200, aspect_ratio=2.0,  resizable='hv', padding=10, ) #, resizable='hv'
        # plot.plot(("freq", "pz"), type="scatter", marker_size=2.0, marker='triangle', color=(0.5,0.3,0.45,0.8), outline_color=(0.1,0.2,0.1,1.0))
        plot.plot(("freq", "sp_S"), type="line", color=(0.72,0.02,0.22,0.85), line_width=3.0)
        plot.plot(("freq", "sp_NS"), type="line", color=(0.0,0.73,0.5,0.85), line_width=3.0)
        # plot.index_axis.title = 'frequency[MHz]'
        plot.value_axis.title = 'strength(abs^2)'

        #add tool
        xbase = np.linspace(0, 1.0, 2001)
        ybase = -1000*np.ones(len(xbase))
        basePlot = create_line_plot((xbase, ybase), color=(1,1,1,0), index_sort="ascending")
        self.rangeSelector =  RangeSelection(basePlot, left_button_selects = True)
        basePlot.active_tool = self.rangeSelector 
        basePlot.overlays.append(RangeSelectionOverlay(component=basePlot))
        plot.add(basePlot)

        self.correlation_spectrum_all = plot
    def _creat_polarPlot(self):
        radius_S = np.array([0,0])
        radius_NS = np.array([0,0])
        theta_S = np.array([0,0])
        theta_NS = np.array([0,0])
        peak_S = create_polar_plot((radius_S,theta_S),color=(0.83,0.12,0.34,0.9), width=5.0)
        peak_NS =  create_polar_plot((radius_NS,theta_NS),color=(0.03,0.80,0.59,0.9), width=5.0)
        plot = Plot(height=200, aspect_ratio=1.0,  resizable='hv',  padding=10, padding_left=5) #, resizable='hv'
        plot.add(peak_S)
        plot.add(peak_NS)
        # plot.index_axis.title = 'complex plane'
        plot.value_axis.title = 'strength(complex)'
        self.polarPlot = plot
    '''     
    def _create_scatterPlot_t1_start(self):
        x = self.tau_f_seq
        y = x*0.0
        plotdata = ArrayPlotData(tau_f=x, spinstateS=y, bright=y, dark=y)
        self.scatterPlot_t1_start_Data = plotdata
        
        plot = Plot(self.scatterPlot_t1_start_Data, width=300, height=180, aspect_ratio=5.0/3.0, resizable='hv', padding=35,)
        plot.plot(("tau_f", "spinstateS"), type="scatter", marker_size=2.0, marker='triangle', color=(0.5,0.3,0.45,0.8), outline_color=(0.1,0.2,0.1,1.0))
        plot.plot(("tau_f", "spinstateS"), type="line", color=(0.9,0.7,0.85,0.8))
        plot.plot(("tau_f", "bright"), type="line", line_width = 2.0,  color=(0.2,0,0,1))
        plot.plot(("tau_f", "dark"), type="line", line_width = 2.0,color=(0,0,0.2,1))

        plot.index_axis.title = 'free evolution[us]'
        plot.value_axis.title = 'spinstate'
        plot.title = "Correlation signal  of t2 at fixed t1(start)"
        self.scatterPlot_t1_start = plot
    
    def _create_scatterPlot_t1_end(self):
        x = self.tau_f_seq
        y = x*0.0
        plotdata = ArrayPlotData(tau_f=x, spinstateS=y, bright=y, dark=y)
        self.scatterPlot_t1_end_Data = plotdata
        
        plot = Plot(self.scatterPlot_t1_end_Data, width=300, height=180, aspect_ratio=5.0/3.0, resizable='hv', padding=35,)
        plot.plot(("tau_f", "spinstateS"), type="scatter", marker_size=2.0, marker='triangle', color=(0.5,0.3,0.45,0.8), outline_color=(0.1,0.2,0.1,1.0))
        plot.plot(("tau_f", "spinstateS"), type="line", color=(0.9,0.7,0.85,0.8))
        plot.plot(("tau_f", "bright"), type="line", line_width = 2.0,  color=(0.2,0,0,1))
        plot.plot(("tau_f", "dark"), type="line", line_width = 2.0,color=(0,0,0.2,1))

        plot.index_axis.title = 'free evolution[us]'
        plot.value_axis.title = 'spinstate'
        plot.title = "Correlation signal  of t2 at fixed t1(end)"
        self.scatterPlot_t1_end = plot    
    def _create_scatterPlot_t2_start(self):
        x = self.tau_f_seq
        y = x*0.0
        plotdata = ArrayPlotData(tau_f=x, spinstateS=y, bright=y, dark=y)
        self.scatterPlot_t2_start_Data = plotdata
        
        plot = Plot(self.scatterPlot_t2_start_Data, width=300, height=180, aspect_ratio=5.0/3.0, resizable='hv', padding=35,)
        plot.plot(("tau_f", "spinstateS"), type="scatter", marker_size=2.0, marker='triangle', color=(0.5,0.3,0.45,0.8), outline_color=(0.1,0.2,0.1,1.0))
        plot.plot(("tau_f", "spinstateS"), type="line", color=(0.9,0.7,0.85,0.8))
        plot.plot(("tau_f", "bright"), type="line", line_width = 2.0,  color=(0.2,0,0,1))
        plot.plot(("tau_f", "dark"), type="line", line_width = 2.0,color=(0,0,0.2,1))

        plot.index_axis.title = 'free evolution[us]'
        plot.value_axis.title = 'spinstate'
        plot.title = "Correlation signal  of t1 at fixed t2(start)"
        self.scatterPlot_t2_start = plot    
    def _create_scatterPlot_t2_end(self):
        x = self.tau_f_seq
        y = x*0.0
        plotdata = ArrayPlotData(tau_f=x, spinstateS=y, bright=y, dark=y)
        self.scatterPlot_t2_end_Data = plotdata
        
        plot = Plot(self.scatterPlot_t2_end_Data, width=300, height=180, aspect_ratio=5.0/3.0, resizable='hv', padding=35,)
        plot.plot(("tau_f", "spinstateS"), type="scatter", marker_size=2.0, marker='triangle', color=(0.5,0.3,0.45,0.8), outline_color=(0.1,0.2,0.1,1.0))
        plot.plot(("tau_f", "spinstateS"), type="line", color=(0.9,0.7,0.85,0.8))
        plot.plot(("tau_f", "bright"), type="line", line_width = 2.0,  color=(0.2,0,0,1))
        plot.plot(("tau_f", "dark"), type="line", line_width = 2.0,color=(0,0,0.2,1))

        plot.index_axis.title = 'free evolution[us]'
        plot.value_axis.title = 'spinstate'
        plot.title = "Correlation signal  of t1 at fixed t2(end)"
        self.scatterPlot_t2_end = plot
    '''  
    def _create_scatterPlot_NS(self):    
        x = self.tau_f_seq
        y = x*0.0
        plotdata = ArrayPlotData(tau_f=x, spinstateNS=y, bright=y, dark=y)
        self.scatterPlot_NSData = plotdata
        plot = Plot(self.scatterPlot_NSData, width=300, height=180, aspect_ratio=5.0/3.0, resizable='hv', padding=35,)
        plot.plot(("tau_f", "spinstateNS"), type="scatter", marker_size=2.0, marker='triangle', color=(0.5,0.7,0.45,0.8), outline_color=(0.1,0.0,0.1,1.0))
        plot.plot(("tau_f", "spinstateNS"), type="line", color=(0.87,0.92,0.85,0.8))

        plot.plot(("tau_f", "bright"), type="line", line_width = 2.0,  color=(0.2,0,0,1))
        plot.plot(("tau_f", "dark"), type="line", line_width = 2.0, color=(0,0,0.2,1))

        plot.index_axis.title = 'free evolution[us]'
        plot.value_axis.title = 'spinstate'
        plot.title = "Non-symmetric correlation signal"
        self.scatterPlot_NS = plot
    '''
    def _create_correlation_spectrum_S(self): 
        n_tau_f = self.tau_f_seq.size # get tau_f number
        plotdata = ArrayPlotData(image=np.zeros((n_tau_f, n_tau_f)))
        self.correlation_spectrum_Sdata = plotdata
        xb = (self.spectrum_FT_freq[1],self.spectrum_FT_freq[-1])
        yb = xb

        plot = Plot(self.correlation_spectrum_Sdata, width=300, height=180, aspect_ratio=1, resizable='hv', padding=35,)
        plot.img_plot(("image"), xbounds=xb, ybounds=yb, colormap=jet, name='image') #how to set xy bounds???
        image = plot.plots['image'][0]
        
        colormap = image.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=plot,
                            orientation='v',
                            resizable='v',
                            width=16,
                            height=80,
                            padding=8,
                            padding_left=10)

        plot.index_axis.title = 'f_1[MHz]'
        plot.value_axis.title = 'f_2[MHz]'
        plot.title = "Symmetric 2d correlation spectrum"

        self.colorbar = colorbar
        self.correlation_spectrum_S = plot
    '''
    def _create_correlation_spectrum_NS(self): 
        n_tau_f = self.tau_f_seq.size # get tau_f number
        plotdata = ArrayPlotData(image=np.zeros((n_tau_f, n_tau_f)))
        self.correlation_spectrum_NSdata = plotdata
        xb = (self.spectrum_FT_freq[1],self.spectrum_FT_freq[-1])
        yb = xb

        plot = Plot(self.correlation_spectrum_NSdata, width=300, height=180, aspect_ratio=1, resizable='hv', padding=35,)
        plot.img_plot(("image"), xbounds=xb, ybounds=yb, colormap=jet) #how to set xy bounds???
        plot.index_axis.title = 'f_1[MHz]'
        plot.value_axis.title = 'f_2[MHz]'
        plot.title = "Non-Symmetric 2d correlation spectrum"

        self.correlation_spectrum_NS = plot
    '''    
    def _create_figureContainer(self):
       # a box contains two signal plots and two correlation spectrum 
       subContainer_top = HPlotContainer(self.correlation_spectrum_S, self.colorbar,  height=260, resizable='h', padding=0)
       subContainert1 = HPlotContainer(self.scatterPlot_t1_start, self.scatterPlot_t1_end, height=270, resizable='h', padding=0)
       subContainert2 = HPlotContainer(self.scatterPlot_t2_start, self.scatterPlot_t2_end, height=270, resizable='h', padding=0)
       # subContainerPolarize = HPlotContainer(self.correlation_spectrum_all, padding=0,)
       self.figureContainer = VPlotContainer(subContainert2, subContainert1,subContainer_top, padding=0)
        

    #==========================================================|
    #     update axis, tool property when relevant changes     |
    #==========================================================|

    @on_trait_change('time_bins, n_laser')
    def update_matrix_plot_axis(self):
        self.matrix_plot.components[0].index.set_data((self.time_bins[0], self.time_bins[-1]), (0.0, float(self.n_laser)))
    @on_trait_change('spectrum_FT_freq')
    def update_correlation_spectrum_S_axis(self):
        freq = self.spectrum_FT_freq
        self.correlation_spectrum_S.components[0].index.set_data((freq[0], freq[-1]), (freq[0], freq[-1]))
    '''
    @on_trait_change('spectrum_FT_freq')
    def update_correlation_spectrum_NS_axis(self):
        freq = self.spectrum_FT_freq
        self.correlation_spectrum_NS.components[0].index.set_data((freq[0], freq[-1]), (freq[0], freq[-1]))
    
    @on_trait_change('rangeSelector.selection')
    def update_selectedRange(self):
        try:
            range_value = self.rangeSelector.selection
            x = self.spectrum_FT_freq 
            xlen = len(x)
            self.selectedRange =  [int(round(range_value[0]*xlen))-1, int(round(range_value[-1]*xlen))-1]
        except:
            self.selectedRange =   self.selectedRange
        # for i in range(len(x)):
        #     if (range_value[0] <= x[i+1]) and (range_value[0] >= x[i]):
        #         self.selectedRange[0] = i
        #         for j in range(i, len(x)):
           #          if range_value[-1] <= x[j]:
           #              self.selectedRange[-1] = j
           #              break
           #          else:
           #              pass
           #      break        
           #  else:
           #      pass
    @on_trait_change('cursor_NS.current_position')
    def update_cursor_S(self):
        self.cursor_S.current_position = self.cursor_NS.current_position
    @on_trait_change('cursor_S.current_position')
    def update_cursor_NS(self):
        self.cursor_NS.current_position = self.cursor_S.current_position
    '''

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

    @on_trait_change('count_data')
    def update_pulse_plot(self):
        self.pulse_plot_data.set_data('y', self.pulse)
        self.pulse_plot_data.set_data('x', self.time_bins)
        self.pulse_plot.components[1].index.set_data(np.array((self.flank, self.flank)))
    '''
    @on_trait_change('peak_S, peak_NS')
    def update_polarPlot(self):
        pS = self.peak_S*self.phase_shift
        pNS = self.peak_NS*self.phase_shift
        self.polarPlot.components[0].index._data = [0.0, pS.real]
        self.polarPlot.components[0].value._data = [0.0, pS.imag]
        self.polarPlot.components[1].index._data = [0.0, pNS.real]
        self.polarPlot.components[1].value._data = [0.0, pNS.imag]
    '''
    @on_trait_change('spin_state')
    def update_scatterPlot_t1_start(self):
        if self.n_ref != 0:
            ss = self.spin_state[:-2*self.n_ref]
        else:
            ss = self.spin_state
        
        n_tau_f = self.tau_f_seq.size # get tau_f number

        y = ss
        y = y.reshape((n_tau_f,n_tau_f))
        ys = y[0,:]

        x = self.tau_f_seq/1000.0 #in us
        
        self.scatterPlot_t1_start_Data.set_data("tau_f", x)
        self.scatterPlot_t1_start_Data.set_data("spinstateS", ys)  


        if self.n_ref != 0 :
            self.brightRef = np.mean(self.spin_state[-2*self.n_ref:-self.n_ref])
            self.darkRef = np.mean(self.spin_state[-self.n_ref:])
            b = [self.brightRef]*len(x)
            d = [self.darkRef]*len(x)
            self.scatterPlot_t1_start_Data.set_data("bright", b)
            self.scatterPlot_t1_start_Data.set_data("dark", d)
        else:
            self.scatterPlot_t1_start_Data.set_data("bright", [np.mean(ys)]*len(x))
            self.scatterPlot_t1_start_Data.set_data("dark", [np.mean(ys)]*len(x))

    @on_trait_change('spin_state')
    def update_scatterPlot_t1_end(self):
        if self.n_ref != 0:
            ss = self.spin_state[:-2*self.n_ref]
        else:
            ss = self.spin_state
        
        n_tau_f = self.tau_f_seq.size # get tau_f number

        y = ss
        y = y.reshape((n_tau_f,n_tau_f))
        ys = y[-1,:]

        x = self.tau_f_seq/1000.0 #in us
        
        self.scatterPlot_t1_end_Data.set_data("tau_f", x)
        self.scatterPlot_t1_end_Data.set_data("spinstateS", ys)  


        if self.n_ref != 0 :
            self.brightRef = np.mean(self.spin_state[-2*self.n_ref:-self.n_ref])
            self.darkRef = np.mean(self.spin_state[-self.n_ref:])
            b = [self.brightRef]*len(x)
            d = [self.darkRef]*len(x)
            self.scatterPlot_t1_end_Data.set_data("bright", b)
            self.scatterPlot_t1_end_Data.set_data("dark", d)
        else:
            self.scatterPlot_t1_end_Data.set_data("bright", [np.mean(ys)]*len(x))
            self.scatterPlot_t1_end_Data.set_data("dark", [np.mean(ys)]*len(x))

    @on_trait_change('spin_state')
    def update_scatterPlot_t2_start(self):
        if self.n_ref != 0:
            ss = self.spin_state[:-2*self.n_ref]
        else:
            ss = self.spin_state
        
        n_tau_f = self.tau_f_seq.size # get tau_f number

        y = ss
        y = y.reshape((n_tau_f,n_tau_f))
        ys = y[:,0]

        x = self.tau_f_seq/1000.0 #in us
        
        self.scatterPlot_t2_start_Data.set_data("tau_f", x)
        self.scatterPlot_t2_start_Data.set_data("spinstateS", ys)  


        if self.n_ref != 0 :
            self.brightRef = np.mean(self.spin_state[-2*self.n_ref:-self.n_ref])
            self.darkRef = np.mean(self.spin_state[-self.n_ref:])
            b = [self.brightRef]*len(x)
            d = [self.darkRef]*len(x)
            self.scatterPlot_t2_start_Data.set_data("bright", b)
            self.scatterPlot_t2_start_Data.set_data("dark", d)
        else:
            self.scatterPlot_t2_start_Data.set_data("bright", [np.mean(ys)]*len(x))
            self.scatterPlot_t2_start_Data.set_data("dark", [np.mean(ys)]*len(x))

    @on_trait_change('spin_state')
    def update_scatterPlot_t2_end(self):
        if self.n_ref != 0:
            ss = self.spin_state[:-2*self.n_ref]
        else:
            ss = self.spin_state
        
        n_tau_f = self.tau_f_seq.size # get tau_f number

        y = ss
        y = y.reshape((n_tau_f,n_tau_f))
        ys = y[:,-1]

        x = self.tau_f_seq/1000.0 #in us
        
        self.scatterPlot_t2_end_Data.set_data("tau_f", x)
        self.scatterPlot_t2_end_Data.set_data("spinstateS", ys)  


        if self.n_ref != 0 :
            self.brightRef = np.mean(self.spin_state[-2*self.n_ref:-self.n_ref])
            self.darkRef = np.mean(self.spin_state[-self.n_ref:])
            b = [self.brightRef]*len(x)
            d = [self.darkRef]*len(x)
            self.scatterPlot_t2_end_Data.set_data("bright", b)
            self.scatterPlot_t2_end_Data.set_data("dark", d)
        else:
            self.scatterPlot_t2_end_Data.set_data("bright", [np.mean(ys)]*len(x))
            self.scatterPlot_t2_end_Data.set_data("dark", [np.mean(ys)]*len(x))

    '''
    @on_trait_change('spectrum_FT_freq, spectrum_FT_S, spectrum_FT_NS')
    def update_correlation_spectrum_all(self):
        freq = self.spectrum_FT_freq
        sp_S = abs(self.spectrum_FT_S)**2
        sp_NS = abs(self.spectrum_FT_NS)**2
          
        self.correlation_spectrum_all_data.set_data("freq", freq)
        self.correlation_spectrum_all_data.set_data("sp_S", sp_S)
        self.correlation_spectrum_all_data.set_data("sp_NS", sp_NS)
        # self.correlation_spectrum_all.components[-1] = create_line_plot((self.correlation_spectrum_all_data["freq"], self.correlation_spectrum_all_data["sp_S"]), index_sort="ascending")
    '''
    @on_trait_change('spectrum_FT_freq, spectrum_FT_S, spectrum_FT_NS, selectedRange, para_S_fit, para_NS_fit')
    def update_correlation_spectrum_zoom(self):
        freq = self.spectrum_FT_freq
        sp_S = abs(self.spectrum_FT_S)
        #sp_NS = abs(self.spectrum_FT_NS)
        self.correlation_spectrum_Sdata.set_data("image", sp_S)
        #self.correlation_spectrum_NSdata.set_data("image", sp_NS)
        '''
        if self.perform_fit:
            freq_zoom_fit = np.linspace(freq_zoom[0], freq_zoom[-1], max([1001, len(freq_zoom)+1]))
            sp_S_zoom_fit = fitting.NLorentzians(*self.para_S_fit)(freq_zoom_fit)
            sp_NS_zoom_fit = fitting.NLorentzians(*self.para_NS_fit)(freq_zoom_fit)
            self.correlation_spectrum_Sdata.set_data("freq_fit", freq_zoom_fit)
            self.correlation_spectrum_Sdata.set_data("sp_fit", sp_S_zoom_fit)
            self.correlation_spectrum_NSdata.set_data("freq_fit", freq_zoom_fit)
            self.correlation_spectrum_NSdata.set_data("sp_fit", sp_NS_zoom_fit)  
        else:
            self.correlation_spectrum_Sdata.set_data("freq_fit", freq_zoom)
            self.correlation_spectrum_Sdata.set_data("sp_fit", sp_S_zoom)
            self.correlation_spectrum_NSdata.set_data("freq_fit", freq_zoom)
            self.correlation_spectrum_NSdata.set_data("sp_fit", sp_NS_zoom) 
        ''' 
        # self.correlation_spectrum_S.y_mapper.range.low = min(list(spns)+list(sps)+list(absspns)+list(abssps))
        # self.correlation_spectrum_S.y_mapper.range.high = max(list(spns)+list(sps)+list(absspns)+list(abssps))
        # self.correlation_spectrum_NS.y_mapper.range.low = min(list(spns)+list(sps)+list(absspns)+list(abssps))
        # self.correlation_spectrum_NS.y_mapper.range.high = max(list(spns)+list(sps)+list(absspns)+list(abssps))
    
    #==========================================================|
    #                   save data and graphs                   |
    #==========================================================|
    def save_color_plot(self, filename):
        self.save_figure(self.figureContainer, filename)
    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    def save_all_figure(self, filename):
        self.save_color_plot(filename+ '_SNSC_Plot.png')
        self.save_matrix_plot(filename+ '_SNSC_Matrix_Plot.png')




    get_set_items = measurePulsed.get_set_items + [ 
                    'thresh_high', 'thresh_low',
                    '__doc__', 
                    'laserTime', 'wait', 'frequency', 'power', 'tau_DD', 't_2pi_y', 't_2pi_x','t_rabi', 
                    'tau_f_start', 'tau_f_end', 'delta_tau_f', 'tau_f_seq', 'n_pi', 'n_ref',
                    'polarizeNspinSSR', 'repetition_SSR', 'laserTime_SSR', 'wait_SSR', 'n_pi_SSR', 'tau_DD_SSR', 't_pi_SSR',
                    'polarizeNspinPuPol', 'repetition_PuPol', 'repetition_P', 'laserTime_PuPol', 'wait_PuPol', 'tau_PuPol', 't_rf2', 'espin0or1', 'reinitespin',
                    'enhancementreadout', 'waitp','waith', 't_pi_rf1', 't_pi_rf2', 'rf1Frequency', 'rf2Frequency', 'rf1Power', 'rf2Power',
                    'integration_width', 'position_signal', 'position_normalize', 'pulse', 'flank','spin_state', 'spin_state_error', 
                    'brightRef', 'darkRef','spectrum_FT_S','spectrum_FT_freq',
                    ]

    traits_view = View(
                       VGroup(
                              VGroup(
                                     HGroup(
                                            Item('submit_button', show_label=False),
                                            Item('remove_button', show_label=False),
                                            Item('resubmit_button', show_label=False),
                                            Item('priority'),
                                            Item('state', style='readonly'),
                                            Item('run_time', style='readonly', format_str='%.f'),
                                            Item('stop_time'),
                                            ),
                                     ),
                              HGroup(
                                      VGroup(
                                             HGroup(
                                                    VGroup(
                                                           Item('bin_width', width= -80, enabled_when='state != "run"'),
                                                           Item('record_length', width= -80, enabled_when='state != "run"'),
                                                           Item('integration_width', width= -80, enabled_when='state != "run"'),
                                                           Item('position_signal', width= -80, enabled_when='state != "run"'),
                                                           Item('position_normalize', width= -80, enabled_when='state != "run"'),
                                                           Item('laserTime', width= -80, enabled_when='state != "run"'),
                                                           Item('wait', width= -80, enabled_when='state != "run"'),
                                                           Item('frequency', width= -80, enabled_when='state != "run"'),
                                                           Item('power', width= -80, enabled_when='state != "run"'),
                                                           ),
                                                    VGroup(
                                                           Item('t_2pi_x', width= -80, enabled_when='state != "run"'),
                                                           Item('t_2pi_y', width= -80, enabled_when='state != "run"'),
                                                           Item('t_rabi', width= -80, enabled_when='state != "run"'),
                                                           Item('n_pi', width= -80, enabled_when='state != "run"'),
                                                           Item('tau_DD', width= -80, enabled_when='state != "run"'),
                                                           Item('delta_tau_f', width= -80, enabled_when='state != "run"'),
                                                           Item('tau_f_start', width= -80, enabled_when='state != "run"'),
                                                           Item('tau_f_end', width= -80, enabled_when='state != "run"'),
                                                           Item('n_ref', width= -80, enabled_when='state != "run"'),
                                                           Item('waith', width= -80, enabled_when='state != "run"'),
                                                           ),
                                                    ),
                                             # HGroup(
                                             #        VGroup(
                                             #               Item('polarizeNspinSSR', width= -80, enabled_when='state != "run"'),
                                             #               Item('repetition_SSR', width= -80, enabled_when='state != "run" and polarizeNspinSSR == True'),
                                             #               Item('laserTime_SSR', width= -80, enabled_when='state != "run" and polarizeNspinSSR == True'),
                                             #               Item('wait_SSR', width= -80, enabled_when='state != "run" and polarizeNspinSSR == True'),
                                             #               ),
                                             #        VGroup(
                                             #               Item('t_pi_SSR', width= -80, enabled_when='state != "run" and polarizeNspinSSR == True'),
                                             #               Item('n_pi_SSR', width= -80, enabled_when='state != "run" and polarizeNspinSSR == True'),
                                             #               Item('tau_DD_SSR', width= -80, enabled_when='state != "run" and polarizeNspinSSR == True'),
                                             #               ),
                                             #        label="SSR",
                                             #        show_border = True,
                                             #        ),
                                             HGroup(
                                                    VGroup(
                                                           Item('polarizeNspinPuPol', width= -80, enabled_when='state != "run"'),
                                                           Item('laserTime_PuPol', width= -80, enabled_when='state != "run" and polarizeNspinPuPol == True'),
                                                           Item('wait_PuPol', width= -80, enabled_when='state != "run" and polarizeNspinPuPol == True'),
                                                           Item('rf2Frequency', width= -80, enabled_when='state != "run" and polarizeNspinPuPol == True'),
                                                           Item('rf2Power', width= -80, enabled_when='state != "run" and polarizeNspinPuPol == True'),
                                                           Item('t_rf2', width= -80, enabled_when='state != "run" and polarizeNspinPuPol == True'),
                                                           ),
                                                    VGroup(
                                                           Item('tau_PuPol', width= -80, enabled_when='state != "run" and polarizeNspinPuPol == True'),
                                                           Item('repetition_PuPol', width= -80, enabled_when='state != "run" and polarizeNspinPuPol == True'),
                                                           Item('repetition_P', width= -80, enabled_when='state != "run" and polarizeNspinPuPol == True'),
                                                           Item('espin0or1', width= -80, enabled_when='state != "run" and polarizeNspinPuPol == True'),
                                                           Item('reinitespin', width= -80, enabled_when='state != "run" and polarizeNspinPuPol == True'),
                                                           ),
                                                    label="PuPol",
                                                    show_border = True,
                                                    ),
                                             # HGroup(
                                             #        VGroup(
                                             #               Item('enhancementreadout', width= -80, enabled_when='state != "run"'),
                                             #               Item('waitp', width= -80, enabled_when='state != "run" and enhancementreadout == True'),
                                             #               Item('t_pi_rf1', width= -80, enabled_when='state != "run" and enhancementreadout == True'),
                                             #               Item('t_pi_rf2', width= -80, enabled_when='state != "run" and enhancementreadout == True'),       
                                             #               Item('total_time', style='readonly',),       
                                                           
                                             #               ),
                                             #        VGroup(  
                                             #               Item('rf1Frequency', width= -80, enabled_when='state != "run" and enhancementreadout == True'),
                                             #               Item('rf1Power', width= -80, enabled_when='state != "run" and enhancementreadout == True'),
                                             #               Item('rf2Frequency', width= -80, enabled_when='state != "run" and enhancementreadout == True'),
                                             #               Item('rf2Power', width= -80, enabled_when='state != "run" and enhancementreadout == True'),
                                             #               ),
                                             #        label="parameters for RE",
                                             #        show_border = True,
                                             #        ),
                                             HGroup(
                                                    Item('resolution', width= -80, style='readonly'),
                                                    Item('freqRange', width= -80, style='readonly'),
                                                    ),
                                             HGroup(
                                                    Item('numDataPoint', width= -80, style='readonly'),
                                                    ),
                                             HGroup(
                                                    Item('thresh_low', width=-80),
                                                    Item('thresh_high', width=-80),
                                                    ),
                                             show_border = True,
                                             ),
                                      VGroup(
                                             Tabbed(
                                                    VGroup(#Item('cursorPosition', style='readonly'),
                                                           Item('figureContainer',editor=ComponentEditor(), show_label=False, height=800, resizable=True),
                                                           label='spectrum',
                                                           ),
                                                    VGroup(
                                                           Item('matrix_plot', editor=ComponentEditor(), show_label=False, height=800, resizable=True),
                                                           Item('show_raw_data'),
                                                           label='raw data',
                                                           ),
                                                    VGroup(
                                                           Item('pulse_plot', editor=ComponentEditor(), show_label=False, height=800, resizable=True),
                                                           label='profile',
                                                           ),
                                                    ),
                                             ),
                                      ),
                              ),
                       menubar=MenuBar(
                                       Menu(
                                            Action(action='load', name='Load'),
                                            Action(action='save', name='Save (.pyd or .pys)'),
                                            Action(action='saveMatrixPlot', name='SaveMatrixPlot (.png)'),
                                            Action(action='saveColorPlot', name='SavePlot (.png)'),
                                            Action(action='saveAll', name='Save All (.png+.pys)'),
                                            Action(action='_on_close', name='Quit'),
                                            name='File'
                                           ),
                                       ),
                       title='Symmetric and Nonsymmetric 2d correlation spectrum XYn',
                       width=1350,  
                       resizable=True,
                       handler=SNSC2dHandler,
                       )