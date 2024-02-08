from tokenize import String
import numpy as np

fft = np.fft.fft
ifft = np.fft.ifft

from hardware.waveform import Sin, IQWaveform, Sequence, Idle
from hardware.waveform610 import Waveform610

from traits.api import Instance, Property, String, Range, Float, Int, Bool, Array, List, Enum, Trait,\
                        Button, on_trait_change, cached_property, Code
from traitsui.api import View, Item, HGroup, VGroup, Tabbed, VSplit, TextEditor#, EnumEditor, RangeEditor, 
from traitsui.menu import Action, Menu, MenuBar
from traitsui.file_dialog import save_file

from enable.api import Component, ComponentEditor
from chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, CMapImagePlot
from chaco.tools.api import PanTool
from chaco.tools.simple_zoom import SimpleZoom

import logging
import time

import analysis.fitting as fitting

from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin, GetSetItemsHandler
from tools.color import scheme

from measurements.pulsed_2smiq_awg_rf import Pulsed
# # matplotlib stuff
# from tools.utility import MPLFigureEditor
# from matplotlib.figure import Figure as MPLFigure
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import IndexLocator

# Handler

# Todo: Consider the stub in waveform and waveform610. Since there is only Pi pulse in NuclearRabi, the phase difference of mw because of the stub in pi mw pulse seems doesn't influence the signal. Be careful if there is any non-pi pulse.



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

class PulsedToolHandler( GetSetItemsHandler ):
    def save_matrix_plot(self, info):
        filename = save_file(title='Save Matrix Plot')
        if filename is '':
            return
        else:
            if filename.find('.png')==-1:
                filename=filename+'.png'
            info.object.save_matrix_plot(filename)
    
    def save_line_plot(self, info):
        filename = save_file(title='Save Line Plot')
        if filename is '':
            return
        else:
            if filename.find('.png')==-1:
                filename=filename+'.png'
            info.object.save_line_plot(filename)

    def save_fft_plot(self, info):
        filename = save_file(title='Save FFT Plot')
        if filename is '':
            return
        else:
            if filename.find('.png')==-1:
                filename=filename+'.png'
            info.object.save_fft_plot(filename)
    
    def save_pulse_plot(self, info):
        filename = save_file(title='Save Pulse Plot')
        if filename is '':
            return
        else:
            if filename.find('.png')==-1:
                filename=filename+'.png'
            info.object.save_pulse_plot(filename)


class Pi_test( Pulsed ):

    """NuclearRabi measurement."""
    """2smiq"""
    """The 1st smiq is for hard Pi pulse and the 2nd smiq is for selective 13 pi pulse"""
    mw_pi   = Range(low=1., high=100000., value=100, desc='length of mw pi pulse [ns]', label='mw_pi [ns]', mode='text', auto_set=False, enter_set=True)
    mw_pi_2 = Range(low=1., high=100000., value=600, desc='length of mw pi pulse of the second smiq[ns]', label='mw_pi 2 [ns]', mode='text', auto_set=False, enter_set=True)

    splitting = Range(low=1, high=20e9,  value=2.160241e6, desc='peak splitting', label='splitting [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))

    """awg610"""
    rf_t   = Range(low=1., high=10000000., value=100000, desc='length of rf pulse [ns]', label='rf_t [ns]', mode='text', auto_set=False, enter_set=True)

    main_dir_rf = String('\NRabi', desc='.WFM/.PAT folder in AWG 610', label='Waveform_rf Dir.')
    freq_awg_rf = Range(low=1, high=2.6e9, value=2.940e6, desc='AWG_rf frequency', label='AWG_rf frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    phase_sin_rf = Range(low=-100.0, high=100.0, value=0.0, desc='Multiple of pi', label='Phase', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    
    amp1_rf  = Range(low=0.0, high=1.0, value=1.0, desc='Waveform_rf amplitude factor', label='wfm amp', auto_set=False, enter_set=True)
    
    

    show_fit = Bool(False, label='show fit')
    
    #fit parameters
    fit_parameters = np.ones(3) #changed from 4 to 3
    rabi_period    = Float(value=0.0, label='T')
    rabi_frequency = Float(value=0.0, label='f')
    rabi_contrast  = Float(value=0.0, label='I')
    rabi_offset    = Float(value=0.0, label='t_0')
    pi2            = Float(value=0.0, label='pi/2')
    pi             = Float(value=0.0, label='pi')
    pi32           = Float(value=0.0, label='3pi/2')
    
    get_set_items = Pulsed.get_set_items + [ 'freq_awg_rf', 'amp1_rf', 'show_fit', 'fit_parameters', 'rabi_frequency', 'rabi_period',
        'rabi_offset', 'rabi_contrast', 'pi2', 'pi', 'pi32',  'mw_pi', 'mw_pi_2',  'rf_t'
    ]
    
    def __init__(self, pulse_generator, time_tagger, microwave, microwave2, awg_rf, **kwargs):
        self.dual = False
        super(Pi_test, self).__init__(pulse_generator, time_tagger, microwave, microwave2, awg_rf, **kwargs)
        self.show_fit = True
        self.frequency2 = self.frequency - self.splitting
        
    
    
    @on_trait_change('frequency')
    def _update_frequency2(self):
        self.frequency2 = self.frequency - self.splitting 



    # Sequence ############################
    def generate_sequence(self):
        laser = self.laser
        wait  = self.wait
        trig_interval_rf = self.trig_interval_rf
        trig_delay_rf = self.trig_delay_rf

        mw_pi = self.mw_pi
        mw_pi_2 = self.mw_pi_2
        rf_t = self.rf_t

        '''
        if mw_pi > mw_pi_2:
            max_time = mw_pi
        else:
            max_time = mw_pi_2
        '''
        max_time = mw_pi + mw_pi_2

        sequence = [
            ([], max_time + trig_interval_rf + trig_delay_rf + rf_t - 100),
            (['aom'], laser),
            (['aom'], (576) * 1e9 /self.sampling_rf + 450 ),
            ([], wait )
        ]

        sequence.append(([], max_time + trig_interval_rf + trig_delay_rf + rf_t)) #consider (['sequence'], 100)
        sequence.append((['laser', 'aom'], laser))
        sequence.append((['aom' ], (576) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
        sequence.append(([], wait))


        sequence.append(([], max_time )) #consider (['sequence'], 100)
        sequence.append((['awg_rf'],  trig_interval_rf))
        sequence.append(([], trig_delay_rf))
        sequence.append(([], rf_t))
        sequence.append((['laser', 'aom'], laser))
        sequence.append((['aom' ], (576) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
        sequence.append(([], wait))

        sequence.append(([], max_time - mw_pi)) #consider (['sequence'], 100)
        sequence.append((['awg_rf'],  trig_interval_rf))
        sequence.append(([], trig_delay_rf))
        sequence.append(([], rf_t))
        sequence.append((['A','mw_x'], mw_pi))
        sequence.append((['laser', 'aom'], laser))
        sequence.append((['aom' ], (576) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
        sequence.append(([], wait))

        sequence.append(([], max_time - mw_pi_2)) #consider (['sequence'], 100)
        sequence.append((['awg_rf'],  trig_interval_rf))
        sequence.append(([], trig_delay_rf))
        sequence.append(([], rf_t))
        sequence.append((['B'], mw_pi_2))
        sequence.append((['laser', 'aom'], laser))
        sequence.append((['aom' ], (576) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
        sequence.append(([], wait))


        if mw_pi > mw_pi_2:
            sequence.append(([], mw_pi_2))
        else:
            sequence.append(([], mw_pi))
        sequence.append((['awg_rf'],  trig_interval_rf))
        sequence.append(([], trig_delay_rf))
        sequence.append(([], rf_t))
        if mw_pi > mw_pi_2:
            sequence.append((['A','mw_x','B'], mw_pi_2))
            sequence.append((['A','mw_x'], mw_pi - mw_pi_2))
        else:
            sequence.append((['A','mw_x','B'], mw_pi))
            sequence.append((['B'], mw_pi_2 - mw_pi))
        sequence.append((['laser', 'aom'], laser))
        sequence.append((['aom' ], (576) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
        sequence.append(([], wait))



        sequence.append((['awg_rf'],  trig_interval_rf))
        sequence.append(([], trig_delay_rf))
        sequence.append(([], rf_t))
        sequence.append((['A','mw_x'], mw_pi))
        sequence.append((['B'], mw_pi_2))
        sequence.append((['laser', 'aom'], laser))
        sequence.append((['aom' ], (576) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
        sequence.append(([], wait))


        sequence.append((['sequence'], 100))
        
        return sequence
    
    def compile_waveforms(self):
        # awg frequency in terms of sampling rate
        #f1 = (self.freq1 - self.freq)/self.sampling

        f1_rf = self.freq_awg_rf/self.sampling_rf
        phase_rf = self.phase_sin_rf*np.pi
        rf_t = self.rf_t * self.sampling_rf / 1e9

        # pulse objects
        mw_rf = Sin(rf_t, freq=f1_rf, amp=self.amp1_rf, phase=phase_rf)
        zero_rf = Idle(1)
        
        # pulse sequence
        pulse_sequence_rf = [zero_rf] + [mw_rf] + [zero_rf]
        
        
        seq_rf=Sequence('NRABI')
        

        name_rf = 'NRABI_pi_rf'
        #waves = IQWaveform(name, pulse_sequence, file_type=0) # 0 for wfm file and 1 for pat file
        wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
        self.waves_rf.append(wave_rf)
        seq_rf.append(wave_rf, wait=True)
            
        self.waves_rf.append(seq_rf)
        self.main_wave_rf = seq_rf.name
    

    
    # Processing ########################
    @on_trait_change('count_data, time_window_width, time_window_offset_signal, time_window_offset_normalize')
    def _analyze_count_data(self):
        y = self.count_data
        n = len(y)
        
        T =  self.time_window_width
        t0 = self.time_window_offset_signal
        t1 = self.time_window_offset_normalize
        
        dt = float(self.bin_width)
        
        if n == 0:
            return None
        
        profile = y.sum(0)
        
        z0 = np.empty((y.shape[0],))
        z1 = np.empty((y.shape[0],))
        
        flank = fitting.find_edge(profile)
        
        i0 = int(np.round((flank + t0) / dt))
        i1 = int(np.round((flank + t1) / dt))
        I  = int(np.round(       T     / dt))
        for i, slot in enumerate(y):
            z0[i] = slot[i0:i0+I].mean()
            z1[i] = slot[i1:i1+I].mean()
        
        self.flank = flank
        self.pulse_profile = profile
        '''
        if self.dual:
            self.y2 = (z0 / z1)[n/2:]
            self.y1 = (z0 / z1)[:n/2]
        else:
            self.y1 = z0 / z1
        '''
        y1 = z0 / z1
        self.y1 = y1
    


    @on_trait_change('y1')
    def _update_line_plot_value(self):
        """override"""
        y = self.y1
        n = len(y)
        old_index = self.line_plot_data.get_data('index')
        if old_index is not None and len(old_index) != n:
            self.line_plot_data.set_data('index', np.arange(n))
        self.line_plot_data.set_data('y1', y)
        if self.show_fit:
            self.line_plot_data.set_data('fit', fitting.Cosinus(*self.fit_parameters)(self.tau))
    
    
    @on_trait_change('show_fit')
    def _plot_fit(self):
        plot = self.line_plot
        if self.show_fit == False:
            while len(plot.components) > 1:
                plot.remove(plot.components[-1])
        else:
            self.line_plot_data.set_data('fit', fitting.Cosinus(*self.fit_parameters)(self.tau))
            plot.plot(('index', 'fit'), style='line', line_width=2, color=scheme['fit 1'])
        plot.request_redraw()
    
    
    @on_trait_change('y1, show_fit')
    def _update_fit_parameters(self):
        if self.y1 is None:
            return
        else:
            y_offset=self.y1.mean()
            
            x = self.tau
            y = self.y1 - y_offset
            
            try:
                #print(fitting.CosinusNoOffsetEstimator)
                #amp, period, phi = fitting.fit(x, y, fitting.CosinusNoOffset, fitting.CosinusNoOffsetEstimator)
                fit_result = fitting.fit_rabi_phase(x, y, np.sqrt(self.y1))
                p, v, q, chisqr = fit_result
                amp, period, phi = p
            except:
                return
            '''
            if amp < 0:
                amp = -amp
                try:
                    #amp, period, phi = fitting.fit(x, y, fitting.CosinusNoOffset, (amp, period, phi))
                    fit_result = fitting.fit_rabi(x, y, fitting.CosinusNoOffset, (amp, period, phi))
                    p, v, q, chisqr = fit_result
                    amp, period, phi = p
                except:
                    return
            
            try:
                amp, period, phi, off = fitting.fit(x, self.y1, fitting.Cosinus, (amp, period, phi, y_offset))
            except:
                return
            while(phi > 0.5 * period):
                phi -= period
            try:
                amp, period, phi, y_offset = fitting.fit(x, self.y1, fitting.Cosinus, (amp, period, phi, y_offset))
            except:
                return
            
            '''
            
            self.fit_parameters = (amp, period, y_offset + phi)
            self.rabi_period    = period
            self.rabi_frequency = 1000.0 / period
            self.rabi_contrast  = 200 * amp / y_offset
            self.rabi_offset    = phi
            self.pi2            = 0.25 * period + phi
            self.pi             = 0.5  * period + phi
            self.pi32           = 0.75 * period + phi
    
    
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
                            Item('mw_pi', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80),
                            Item('splitting', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                            Item('frequency2', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.6f GHz' % (x*1e-9))),
                            Item('power2', width=-40),
                            Item('mw_pi_2', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80),
                            label='SMIQ',
                            show_border=True,
                        ),
                        HGroup(
                            Item('reload_awg_rf'),
                            Item('wfm_button_rf', show_label=False),
                            Item('upload_progress_rf', style='readonly', format_str='%i'),
                            Item('main_dir_rf', style='readonly'),
                            label='AWG',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        #Item('freq_tot', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.4f GHz' % (x*1e-9))),
                        Item('freq_awg_rf', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                        Item('phase_sin_rf', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80),
                        Item('amp1_rf', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80),
                        Item('rf_t', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%.2f'%x), width=-80),
                        label='Waveform_rf',
                        show_border=True,
                    ),
                    HGroup(
                        Item('show_fit'),
                        Item('rabi_contrast', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.1f% %'%x)),
                        Item('rabi_frequency', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f MHz'%x)),
                        Item('rabi_period', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f ns'%x)),
                        Item('rabi_offset', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f ns'%x)),
                        Item('pi2', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f ns'%x)),
                        Item('pi', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f ns'%x)),
                        Item('pi32', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f ns'%x)),
                        label='Fit',
                        show_border=True,
                    ),
                    VGroup(
                        Item('line_plot', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                        HGroup(
                            Item('fft_plot', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),
                            Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=500, height=300, resizable=True),                    
                        )
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
        title='Pi test',
        buttons=[],
        resizable=True,
        width =1400,
        height=800,
        handler=PulsedToolHandler,
        menubar=MenuBar(
            Menu(
                Action(action='load', name='Load'),
                Action(action='save', name='Save (.pyd or .pys)'),
                Action(action='saveLinePlot', name='SaveLinePlot (.png)'),
                Action(action='saveMatrixPlot', name='SaveMatrixPlot (.png)'),
                Action(action='saveColorPlot', name='SavePlot (.png)'),
                Action(action='saveAll', name='Save All (.png+.pys)'),
                Action(action='_on_close', name='Quit'),
                name='File',
            ),
        ),
    )
    
 