import numpy as np

fft = np.fft.fft
ifft = np.fft.ifft

from hardware.waveform import *

from traits.api import Instance, Property, Range, Float, Int, String, Bool, Array, List, Enum, Trait,\
                                 Button, on_trait_change, cached_property, Code
from traitsui.api import View, Item, HGroup, VGroup, Tabbed, VSplit, TextEditor#, EnumEditor, RangeEditor, 
from traitsui.menu import Action, Menu, MenuBar
from traitsui.file_dialog import save_file

from enable.api import Component, ComponentEditor
from chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, PlotLabel
from chaco.tools.api import PanTool
from chaco.tools.simple_zoom import SimpleZoom

import logging
import time

import analysis.fitting as fitting

from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin, GetSetItemsHandler
from tools.color import scheme
from pulsed_awg import Pulsed, PulsedToolHandler, find_laser_pulses




class ODMR( Pulsed ):
    """Provides ODMR measurements via AWG device. Pulsed only"""
    
    # parameters
    t_pi = Range(low=1., high=1.0e9, value=1700., desc='length of pi pulse [ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    amp1   = Range(low=0.0, high=1.0,     value=1.0,    desc='Waveform amplitude factor',  label='wfm amp',        mode='text', auto_set=False, enter_set=True)
    frequency_begin_p = Range(low=1, high=12.3e9, value=2.86e9, desc='Start Frequency Pmode[Hz]', label='Begin f[Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    frequency_end_p = Range(low=1, high=12.3e9, value=2.8815e9, desc='Stop Frequency Pmode[Hz]', label='End f[Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    frequency_delta_p = Range(low=1e-3, high=3.3e9, value=1.0e5, desc='frequency step Pmode[Hz]', label='Delta f[Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    Hermite_pulse = Bool(False, label='Hermite_pulse')
    Hermite_c = Range(Low=0.0, high=3.0, value=0.956, desc='c_para', label='Hermite_c_para', auto_set=False, enter_set=True)

    # control data fitting
    show_fit = Bool(False, label='show fit')
    number_of_resonances = Trait('auto', String('auto', auto_set=False, enter_set=True), Int(10000., desc='Number of Lorentzians used in fit', label='N', auto_set=False, enter_set=True))
    threshold = Range(low= -99, high=99., value= -50., desc='Threshold for detection of resonances [%]. The sign of the threshold specifies whether the resonances are negative or positive.', label='threshold [%]', mode='slider', auto_set=False, enter_set=True)
    
    # fit result
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    fit_frequencies = Array(value=np.array((np.nan,)), label='frequency [Hz]') 
    fit_line_width = Array(value=np.array((np.nan,)), label='line_width [Hz]') 
    fit_contrast = Array(value=np.array((np.nan,)), label='contrast [%]')

    # sweep index
    frequency = Array()

    
    get_set_items = Pulsed.get_set_items + ['t_pi', 'amp1', 
                                            'frequency_begin_p', 'frequency_end_p', 'frequency_delta_p', 
                                            'show_fit', 'number_of_resonances', 'threshold', 
                                            'fit_parameters', 'fit_frequencies', 'fit_line_width', 'fit_contrast',
                                            'frequency', 'Hermite_pulse', 'Hermite_c']
    
    def __init__(self, pulse_generator, time_tagger, microwave, awg, **kwargs):
        self.dual = False
        super(ODMR, self).__init__(pulse_generator, time_tagger, microwave, awg, **kwargs)
        self.show_fit = False
        self.frequency = np.arange(self.frequency_begin_p, self.frequency_end_p, self.frequency_delta_p)
        self.modify_plot()

    def prepare_awg(self):
        if self.reload_awg:
            self.load_wfm()
        self.awg.set_vpp(self.vpp1, 0b01)
        self.awg.set_vpp(self.vpp2, 0b10)
        self.awg.set_sampling(self.sampling)
        self.awg.set_mode('E')
        
    def load_wfm(self):
        self.waves = []
        self.main_wave = ''
        self.frequency = np.arange(self.frequency_begin_p, self.frequency_end_p, self.frequency_delta_p)
        self.compile_waveforms()
        print("uploading waveform...........................................................")
        self.awg.upload(self.waves)
        self.awg.managed_load(self.main_wave)
        self.reload_awg = False

    # Sequence ############################
    def generate_sequence(self):
        t_pi = self.t_pi
        laser = self.laser
        wait  = self.wait
        trig_interval = self.trig_interval
        sequence = [ (['green'], laser),
                     ([ ]      , wait)
                   ]
        for i in range(len(self.frequency)):
            sequence.append( (['awg']            , trig_interval       ) )
            sequence.append( ([]                 , t_pi + trig_interval) )
            sequence.append( (['laser', 'green'], laser               ) )
            sequence.append( ([ ]                , wait                ) )
        sequence.append( (['sequence'], 100) )
        return sequence
    
    def compile_waveforms(self):
        # awg frequency, duration in terms of sampling rate
        frequency = self.frequency/self.sampling
        t_pi = self.t_pi * self.sampling / 1e9 # time in terms of point

        # pulse objects
        mw_sin = Sin(duration=t_pi, freq=1.0, amp=self.amp1, phase=0)
        if self.Hermite_pulse:
            envelope = HermiteEnve(t_pi, c_para=self.Hermite_c)
            mw = mw_sin*envelope 
        else:
            mw = mw_sin
        zero = Idle(1)
        # pulse sequence
        pulse_sequence = [zero, mw, zero]
        idleforminpointrequirement = Idle(256) # a waveform contain min. 2400 points
        pulse_sequence = pulse_sequence + [idleforminpointrequirement]


        seqName = ('ODMR')
        seq = Sequence(seqName)
        for i,freq in enumerate(frequency):
            mw_sin.freq = freq
            mw.freq = freq # modify freq
            name = 'ODMR_%03i' % i
            assetType = "Waveform"
            waves = IQWaveform(name, pulse_sequence, file_type=0)
            self.waves.append(waves[0])
            self.waves.append(waves[1])
            seq.append(waves, wait=True)

        # a list containing a SQEX object, it's kind of dummy but just to fit the original code
        self.waves.append(seq)
        # seq.name is different from seqName, 
        # seq.name contained file extension (*.seqx)
        self.main_wave = seq.name # specify the main seq, in this rabi case only one seq
   
    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        frequency = np.arange(self.frequency_begin_p, self.frequency_end_p, self.frequency_delta_p)
        n_bins = int(self.record_length / self.bin_width)
        time_bins = self.bin_width*np.arange(n_bins)
        sequence = self.generate_sequence()
        depth = find_laser_pulses(sequence)
        if self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins) and np.any(frequency == self.frequency): # if the sequence and time_bins are the same as previous, keep existing data
            self.old_count_data = self.count_data.copy()
        else:
            self.run_time = 0.0
            self.old_count_data = np.zeros((len(self.frequency),n_bins))
            
        # prepare awg
        if not self.keep_data or np.any(frequency != self.frequency):
            self.prepare_awg()  #TODO: if reload: block thread untill upload is complete
            self.frequency = frequency
            self.run_time = 0.0

        self.sequence = sequence
        self.depth = depth
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
        
    
    # Plot and Fit ########################
    def modify_plot(self):
        self.line_plot.index_axis.title = 'freq [GHz]'

    @on_trait_change('count_data')
    def _update_line_plot_index(self):
        self.line_plot_data.set_data('index', self.frequency/1e9)

    def _create_fft_plot(self):
        pass
    
    def _update_fft_plot_value(self):
        pass

    def _update_fft_axis_title(self):
        pass
        
    @on_trait_change('y1')
    def _update_line_plot_value(self):
        """override"""
        y = self.y1
        n = len(y)
        old_index = self.line_plot_data.get_data('index')
        if old_index is not None and len(old_index) != n:
            self.line_plot_data.set_data('index', self.frequency/1e9)
        self.line_plot_data.set_data('y1', y)
        if self.show_fit:
            self._update_line_plot_data_fit()
    
    @on_trait_change('show_fit')
    def _plot_fit(self):
        plot = self.line_plot
        if self.show_fit == False:
            plot.delplot('fit')
            self.line_label.visible = False
        else:
            self.line_plot_data.set_data('index', self.frequency)
            self._update_line_plot_data_fit()
            plot.plot(('index', 'fit'), style='line', line_width=2, color=scheme['fit 1'], name='fit')
            self.line_label.visible = True
        plot.request_redraw()

    def _update_line_plot_data_fit(self):
        if not np.isnan(self.fit_parameters[0]):            
            self.line_plot_data.set_data('fit', fitting.NLorentzians(*self.fit_parameters)(self.frequency))
            p = self.fit_parameters
            f = p[1::3]
            w = p[2::3]
            N = len(p) / 3
            contrast = np.empty(N)
            c = p[0]
            pp = p[1:].reshape((N, 3))
            for i, pi in enumerate(pp):
                a = pi[2]
                g = pi[1]
                A = np.abs(a / (np.pi * g))
                if a > 0:
                    contrast[i] = 100 * A / (A + c)
                else:
                    contrast[i] = 100 * A / c
            s = ''
            for i, fi in enumerate(f):
                s += 'f %i: %.6e Hz, HWHM %.3e Hz, contrast %.1f%%\n' % (i + 1, fi, w[i], contrast[i])
            self.line_label.text = s


    # fitting
    @on_trait_change('y1,show_fit, number_of_resonances, threshold')
    def _update_fit(self):
        if self.show_fit:
            N = self.number_of_resonances 
            if N != 'auto':
                N = int(N)
            try:
                p = fitting.fit_multiple_lorentzians(self.frequency, self.y1, N, threshold=self.threshold * 0.01)
            except Exception:
                logging.getLogger().debug('ODMR fit failed.', exc_info=True)
                p = np.nan * np.empty(4)
        else:
            p = np.nan * np.empty(4)
        self.fit_parameters = p
        self.fit_frequencies = p[1::3]
        self.fit_line_width = p[2::3]
        N = len(p) / 3
        contrast = np.empty(N)
        c = p[0]
        pp = p[1:].reshape((N, 3))
        for i, pi in enumerate(pp):
            a = pi[2]
            g = pi[1]
            A = np.abs(a / (np.pi * g))
            if a > 0:
                contrast[i] = 100 * A / (A + c)
            else:
                contrast[i] = 100 * A / c
        self.fit_contrast = contrast
    
    # traits_view = View( title='ODMR (AWG)',)
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
                            Item('reload_awg', enabled_when='state == "idle"'),
                            Item('wfm_button', show_label=False, enabled_when='state == "idle"'),
                            Item('upload_progress', style='readonly', format_str='%i'),
                            label='AWG',
                            show_border=True,
                        ),
                        HGroup(
                            Item('freq', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                            Item('power', width=-40),
                            label='SMIQ',
                            show_border=True,
                        ),
                    ),
                    HGroup( 
                        VGroup(    
                            HGroup(
                                Item('t_pi', width=-80, enabled_when='state != "run"'),
                                Item('amp1', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f'%x), width=-80, enabled_when='state != "run"'),
                                label='Transition',
                                show_border=True,
                            ),
                            HGroup(
                                Item('Hermite_pulse'),
                                Item('Hermite_c', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.5f'%x), width=-80),
                            ),
                        ),
                        HGroup(
                            Item('frequency_begin_p', width= -80, enabled_when='state == "idle"'),
                            Item('frequency_end_p', width= -80, enabled_when='state == "idle"'),
                            Item('frequency_delta_p', width= -80, enabled_when='state == "idle"'),
                            label='Sweep',
                            show_border=True,
                        ),
                    ),
                    VGroup( 
                        HGroup(
                            Item('show_fit'),
                            Item('number_of_resonances', width= -60),
                            Item('threshold', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width= -60),
                            Item('fit_contrast', style='readonly'),
                            Item('fit_line_width', style='readonly'),
                            Item('fit_frequencies', style='readonly'),
                        ),
                        label='Fit',
                        show_border=True,
                    ),
                    Tabbed(
                        VGroup(
                            #Item('cursorPosition', style='readonly'),
                            Item('line_plot', editor=ComponentEditor(), show_label=False, width=500, height=400, resizable=True),
                            label='LINE',
                        ),
                        VGroup(
                            Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=500, height=400, resizable=True),
                            label='COUNT',
                        ),
                    ),
                    label = 'data',
                ),
                VGroup( 
                    HGroup( 
                        HGroup(
                            Item('laser', width=-50, enabled_when='state == "idle"'),
                            Item('wait', width=-50, enabled_when='state == "idle"'),
                            label='Sequence',
                            show_border=True,
                        ),
                        HGroup(
                            Item('vpp1', width=-50, enabled_when='state == "idle"'),
                            Item('vpp2', width=-50, enabled_when='state == "idle"'),
                            Item('sampling', width=-80, enabled_when='state == "idle"'),
                            Item('trig_interval', width=-50, enabled_when='state == "idle"'),
                            label='AWG',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item('record_length'),
                        Item('bin_width'),
                        Item('time_window_width'),
                        Item('time_window_offset_signal'),
                        Item('time_window_offset_normalize'),
                        # Item('dual', enabled_when='state == "idle"'),s
                        label='Analysis',
                        show_border=True,
                    ),
                    Item('pulse_plot', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                    label = 'settings',
                ),
            ),
        ),
        menubar=MenuBar(
            Menu(
                Action(action='save', name='Save (.pyd or .pys)'),
                Action(action='load', name='Load (.pyd or .pys)'),
                Action(action='save_matrix_plot', name='Save Matrix Plot (.png)'),
                Action(action='save_line_plot', name='Save Line Plot (.png)'),
                Action(action='save_fft_plot', name='Save FFT Plot (.png)'),
                Action(action='save_pulse_plot', name='Save Pul Plot (.png)'),
                Action(action='save_all', name='Save All(.png) + .pys'),
                Action(action='_on_close', name='Quit'),
                name='File'
            ),
        ),
        title='ODMR (AWG)',
        buttons=[],
        resizable=True,
        width =-900,
        # height=-800,
        handler=PulsedToolHandler,
    )