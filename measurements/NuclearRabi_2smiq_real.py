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


class NuclearRabi_2smiq_real( Pulsed ):

    """NuclearRabi measurement."""

    reload_awg = Bool(False, label='reload', desc='Compile waveforms upon start up.')
    
    """awg610"""
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
    
    get_set_items = Pulsed.get_set_items + [ 'freq_awg_rf',  'show_fit', 'fit_parameters', 'rabi_frequency', 'rabi_period',
        'rabi_offset', 'rabi_contrast', 'pi2', 'pi', 'pi32', 
    ]
    
    def __init__(self, pulse_generator, time_tagger, microwave, microwave2,  awg_rf, **kwargs):
        self.dual = False
        super(NuclearRabi_2smiq_real, self).__init__(pulse_generator, time_tagger, microwave,  microwave2, awg_rf, **kwargs)
        self.show_fit = True
        self.reload_awg = False
        
    
    # override
    def start_up(self):
        """Override for additional stuff to be executed at start up."""
        self.pulse_generator.Night()
        self.awg_rf.set_output(1)
        self.awg_rf.run()

    
    
    
    def shut_down(self):
        """Override for additional stuff to be executed at shut down."""
        self.pulse_generator.Light()
        self.awg_rf.stop()
        self.awg_rf.set_output(0)
    
    
    def shut_down_finally(self):
        """Override for additional stuff to be executed finally."""
        self.pulse_generator.Light()
        self.awg_rf.stop()
        self.awg_rf.set_output(0)


    
    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        self.tau = np.arange(self.start_time, self.end_time, self.time_step)
        n_bins = int(self.record_length / self.bin_width)
        time_bins = self.bin_width*np.arange(n_bins)
        sequence = self.generate_sequence()
        
        self.depth = find_laser_pulses(self.sequence)
        
        if self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins): # if the sequence and time_bins are the same as previous, keep existing data
            self.old_count_data = self.count_data.copy()
        else:
            self.run_time = 0.0
            self.old_count_data = np.zeros((self.depth,n_bins))
            
        # prepare awg
        if not self.keep_data:
            self.prepare_awg_rf()  #TODO: if reload: block thread untill upload is complete
        
        self.sequence = sequence
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
        
    
    def _run(self):
        #new style TODO: include sweeps + abort @ sweep
        try:
            self.state = 'run'
            self.pulse_generator.Night()
            
            self.apply_parameters()
            
            if self.run_time >= self.stop_time:
                logging.getLogger().debug('Runtime larger than stop_time. Returning')
                self.state='done'
                return
            self.start_up()
            
            # set up counters and pulse generator
            if self.dual: self.depth *= 2
            if self.channel_apd_0 > -1:
                pulsed_0 = self.time_tagger.Pulsed(self.n_bins, int(np.round(self.bin_width*1000)), self.depth, self.channel_apd_0, self.channel_detect, self.channel_sequence)
            if self.channel_apd_1 > -1:
                pulsed_1 = self.time_tagger.Pulsed(self.n_bins, int(np.round(self.bin_width*1000)), self.depth, self.channel_apd_1, self.channel_detect, self.channel_sequence)
            self.pulse_generator.Sequence(self.sequence)
            self.pulse_generator.checkUnderflow()
            
            # count
            while self.run_time < self.stop_time:
                start_time = time.time()
                self.thread.stop_request.wait(1.0)
                if self.thread.stop_request.isSet():
                    logging.getLogger().debug('Caught stop signal. Exiting.')
                    break
                if self.pulse_generator.checkUnderflow():
                    raise RuntimeError('Underflow in pulse generator.')
                if self.channel_apd_0 > -1 and self.channel_apd_1 > -1:
                    self.count_data = self.old_count_data + pulsed_0.getData() + pulsed_1.getData()
                elif self.channel_apd_0 > -1:
                    self.count_data = self.old_count_data + pulsed_0.getData()
                elif self.channel_apd_1 > -1:
                    self.count_data = self.old_count_data + pulsed_1.getData()
                self.run_time += time.time() - start_time
            
            # post operation
            self.shut_down()
            self.pulse_generator.Light()
            
        finally:
            self.shut_down_finally()
            self.state='idle'
    

    # Sequence ############################
    def generate_sequence(self):
        tau   = self.tau
        laser = self.laser
        wait  = self.wait
        trig_interval_rf = self.trig_interval_rf
        trig_delay_rf = self.trig_delay_rf


        sequence = [
            (['aom'], laser),
            (['aom'], (576+512) * 1e9 /self.sampling_rf + 450 ), #to be the same with other pulse
            ([ ], wait )
        ]
        
        for t in tau:
            #if t == tau[0]:
            #    sequence.append((['ch2','awg_rf'], trig_interval_rf))
            #else:
            #    sequence.append((['awg_rf'], trig_interval_rf))
            if t == tau[0]:
                sequence.append(([], tau[-1] - t - 100))
            else:
                sequence.append(([ ], tau[-1] - t))
            sequence.append((['awg_rf'], trig_interval_rf))
            sequence.append(([], trig_delay_rf + t))
            sequence.append((['aom', 'laser'], laser))
            sequence.append((['aom'], (576+512) * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610 and excite nulcear
            sequence.append(([ ], wait))
        sequence.append((['sequence'], 100))
        '''
        for t in tau:
            if t == tau[0]:
                sequence.append((['rf','awg'], trig_interval))
            else:
                sequence.append((['awg'], trig_interval))
            sequence.append(([], trig_delay + mw_pi))
            sequence.append((['awg_rf'], trig_interval_rf))
            sequence.append(([], trig_delay_rf + t))
            sequence.append(([ ], 576 * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610
            sequence.append((['awg'], trig_interval))
            sequence.append(([], trig_delay + mw_pi))
            sequence.append((['aom', 'laser'], laser))
            sequence.append(([ ], wait))
        sequence.append((['sequence'], 100))
        '''
        '''
        for t in tau:
            if t == tau[0]:
                sequence.append((['rf','awg_rf'], trig_interval_rf))
                sequence.append(([], trig_delay_rf + t))
            else:
                sequence.append((['awg_rf'], trig_interval_rf+500))
                sequence.append(([], trig_delay_rf + t))
            sequence.append(([ ], 24000))
        sequence.append((['sequence'], 100))
        '''
        return sequence
    
    def compile_waveforms(self):
        # awg frequency in terms of sampling rate
        #f1 = (self.freq1 - self.freq)/self.sampling

        f1_rf = self.freq_awg_rf/self.sampling_rf
        phase_rf = self.phase_sin_rf*np.pi

        # pulse objects
        mw_rf = Sin(0, freq=f1_rf, amp=self.amp1_rf, phase=phase_rf)
        zero_rf = Idle(1)
        
        # pulse sequence
        pulse_sequence_rf = [zero_rf, mw_rf, zero_rf]
        
        seq=Sequence('NRABI')
        seq_rf=Sequence('NRABI')
        

            
        for i,t in enumerate(self.tau):

            t = t*self.sampling_rf/1e9
            mw_rf.duration = t
            name_rf = 'NRABI_%03i' % i
            wave_rf = Waveform610(name_rf, pulse_sequence_rf, offset=0, file_type=0, sampling=self.sampling_rf)# 0 for wfm file and 1 for pat file
            self.waves_rf.append(wave_rf)
            seq_rf.append(wave_rf, wait=True)

            
        self.waves_rf.append(seq_rf)
        self.main_wave_rf = seq_rf.name
    

    
    # Plot and Fit ########################


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
                        Item('start_time', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                        Item('end_time', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                        Item('time_step', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
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
        title='NuclearRabi 2smiq real (AWG610)',
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
    
 