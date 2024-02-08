import numpy as np

fft = np.fft.fft
ifft = np.fft.ifft

from hardware.waveform import *

from traits.api import Instance, Property, Range, Float, Int, Bool, Array, List, Enum, Trait,\
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
from pulsed_awg import Pulsed, PulsedToolHandler



class Tomo( Pulsed ):
    """Random Pulse Tomography"""
    
    freq_awg = Range(low=1, high=1e9, value=40e6, desc='AWG frequency', label='AWG frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    freq_tot = Range(low=1, high=20e9, value=2.87e9, desc='Total frequency', label='Total frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    detuning = Range(low=-50e6, high=50e6, value=1.0e6, desc='AWG frequency', label='Detuning [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    amp1  = Range(low=0.0, high=1.0, value=1.0, desc='Waveform amplitude factor', label='wfm amp', auto_set=False, enter_set=True)
    t_pi = Range(low=0.0, high=2000, value=200., desc='duration of pi pulse', label='pi duration', auto_set=False, enter_set=True)
    t_x = Range(low=0.0, high=2000, value=0., desc='duration of x pulse', label='x duration', auto_set=False, enter_set=True)
    t_y = Range(low=0.0, high=2000, value=0., desc='duration of y pulse', label='y duration', auto_set=False, enter_set=True)
    
    get_set_items = Pulsed.get_set_items + [
        'freq_tot', 'freq_awg', 'detuning', 'amp1', 't_pi'
    ]

    #parameters for f(x)
    K = Range(low=0, high=100, value=10, desc='Number of Fourier components', label='K', mode='text', auto_set=False, enter_set=True)
    n_envelop = Range(low=0, high=100, value=2, desc='Dimension of the system', label='number of envelop', mode='text', auto_set=False, enter_set=True)
    rseed = Int(low=0, high=1000, value=0, desc='Random Seed', label='Seed', mode='text',auto_set=False, enter_set=True)

    amp_f = Array()
    freq_f = Array()
    phase_f = Array()
    
    get_set_items += [
        'K', 'n_envelop', 'rseed',
        'amp_f','freq_f','phase_f',
    ]
    
    def __init__(self, pulse_generator, time_tagger, microwave, awg, **kwargs):
        self.dual = False
        super(Tomo, self).__init__(pulse_generator, time_tagger, microwave, awg, **kwargs)
        self.freq_tot = self.freq + self.freq_awg
        
        r = int(self.rseed)
        K = int(self.K)

        np.random.seed(r)
        self.amp_f = np.random.rand(self.n_envelop,K)
        self.amp_f /= self.amp_f.sum(axis=1)[:,None]  # Normalization
        self.freq_f = np.random.rand(self.n_envelop,K)
        self.phase_f = np.random.rand(self.n_envelop,K)
    

    # Sequence ############################
    def generate_sequence(self):
        tau   = self.tau
        laser = self.laser
        wait  = self.wait
        t_pi  = self.t_pi
        t_x   = self.t_x
        t_y   = self.t_y
        trig_interval = self.trig_interval

        sequence = [
            (['aom'], laser),
            ([ ]      , wait )
        ]
        for i in range(self.n_envelop):
            for t in tau:
                sequence.append((['awg', 'mw_y'], trig_interval))
                sequence.append((['mw_y'], t + t_x + t_y + trig_interval))
                sequence.append((['laser', 'green'] , laser  ) )               
                sequence.append(([], wait))
        sequence.append((['awg','mw_y'], trig_interval))      #reference pi-pulse?
        sequence.append((['mw_y'], t_pi + trig_interval))
        sequence.append((['laser', 'green'], laser))               #-1 state readout
        sequence.append(([], wait))
        sequence.append((['laser', 'green'], laser))               #0 state readout
        sequence.append(([], wait))  
        sequence.append((['sequence'], 100))
        return sequence
    
    def compile_waveforms(self):
        # awg frequency in terms of sampling rate
        # Detuned AWG frequency
        f_det = (self.freq_awg + self.detuning)/self.sampling
        # Resonance AWG frequency
        f_res = self.freq_awg / self.sampling
        
        tau = self.tau * self.sampling / 1e9
        t_x = self.t_x * self.sampling / 1e9
        t_y = self.t_y * self.sampling / 1e9
        ts  = self.t_pi * self.sampling / 1e9

        # pulse objects
        operatorx = Sin(0, freq=f_res, amp=self.amp1, phase=0)
        operatory = Sin(t_y, freq=f_res, amp=self.amp1, phase=np.pi/2.0)
        sine = Sin(0, freq=f_det, amp=self.amp1, phase=0)
        ref = Sin(ts, freq=f_res, amp=self.amp1, phase=0)
        zero = Idle(1)
        idleforminpointrequirement = Idle(256) # a waveform contain min. 2400 points
        
        # pulse sequence        
        
        seq=Sequence('TOMO')

        for p in range(self.n_envelop):
            ffx = 4.0e6 * (self.freq_f[p])/self.sampling
            envelop = Envelop(duration=0, amp_li=self.amp_f[p], freq_li=ffx, phase_li=self.phase_f[p])
            # mw = sine*envelop
            mw = envelop

            pulse_sequence = [zero, operatorx, operatory , mw, zero] + [idleforminpointrequirement]

            for i,t in enumerate(tau):
                # increment microwave duration 
                mw.duration = t
                name = 'FX_%02i_%03i' % (p,i)
                # file_type=0 -> .WFM
                # file_type=1 -> .PAT
                waves = IQWaveform(name, pulse_sequence, file_type=0)
                self.waves.append(waves[0])
                self.waves.append(waves[1])
                seq.append(waves, wait=True)

        pulse_sequencer = [zero, ref, zero] + [idleforminpointrequirement]
        namer = 'REF'
        waver = IQWaveform(namer, pulse_sequencer, file_type=0)
        self.waves.append(waver[0])
        self.waves.append(waver[1])
        seq.append(waver, wait=True)

        self.waves.append(seq)
        self.main_wave = seq.name
    
    
    # Plot and Fit ########################
    def _create_line_plot(self):
        Y = {
            'index':np.array((0,1)),
            'bright':np.array((1, 1)),
            'dark':np.array((0, 0))
        }
        for i in range(self.n_envelop):
            Y['y%d' % i] = np.array((0,0))
        line_plot_data = ArrayPlotData(**Y)     # Unpack all the items in Y as kwargs

        plot = Plot(line_plot_data, padding=8, padding_left=64, padding_bottom=36)
        for i in range(self.n_envelop):
            plot.plot(('index','y%d' % i), color='auto', line_width=2, id='0', name='y%d' % i)
        
        plot.plot(('index', 'bright'), color='black', name='bright')
        plot.plot(('index', 'dark'), color='black', name='dark')
        
        plot.bgcolor = scheme['background']
        plot.x_grid = None
        plot.y_grid = None
        plot.index_axis.title = 'tau [ns]'
        plot.value_axis.title = 'intensity [a.u.]'
        line_label = PlotLabel(text='', hjustify='left', vjustify='bottom', position=[64, 128])
        plot.overlays.append(line_label)
        # plot.tools.append(PanTool(plot))
        # plot.overlays.append(SimpleZoom(plot, enable_wheel=False))
        self.line_label = line_label
        self.line_plot_data = line_plot_data
        self.line_plot = plot

    @on_trait_change('y1')
    def _update_line_plot_value(self):
        """override"""
        y = self.y1
        Y_data = []
        n_data = len(y) - 2     # Number of data point except the references
        n_index = int((n_data)/self.n_envelop)
        for i in range(self.n_envelop):
            head = int(n_index*i)      # i starts from 0
            tail = int(n_index*(i+1))
            Y_data.append(y[head:tail])
        bright = y[-2:-1]
        dark = y[-1:]
        old_index = self.line_plot_data.get_data('index')

        if old_index is not None and len(old_index) != n_index:
            self.line_plot_data.set_data('index', self.tau)
        for i in range(self.n_envelop):
            self.line_plot_data.set_data('y%d' % i, Y_data[i])
        self.line_plot_data.set_data('bright', bright * np.ones(n_index))
        self.line_plot_data.set_data('dark', dark * np.ones(n_index))
    
    @on_trait_change('freq, freq_awg')
    def _update_freq_tot(self):
        self.freq_tot = self.freq + self.freq_awg
    
    @on_trait_change('show_fit')
    def _plot_fit(self):
        plot = self.line_plot
    
    @on_trait_change('n_envelop, rseed, K')
    def _freq_value(self):
        np.random.seed(self.rseed)
        self.amp_f = np.random.rand(self.n_envelop,self.K)
        self.amp_f /= self.amp_f.sum(axis=1)[:,None]  # Normalization
        self.freq_f = np.random.rand(self.n_envelop,self.K)
        self.phase_f = np.random.rand(self.n_envelop,self.K)
        self._create_line_plot()
    
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
                        Item('freq_tot', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.4f GHz' % (x*1e-9))),
                        Item('freq_awg', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                        Item('detuning', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                        Item('amp1', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.5f'%x), width=-80),
                        label='Frequency',
                        show_border=True,
                    ),
                    HGroup(
                        Item('t_pi', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f'%x), width=-80),
                        Item('t_x', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f'%x), width=-80),
                        Item('t_y', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f'%x), width=-80),
                        label='Initialization',
                        show_border=True,
                    ),
                    HGroup(
                        HGroup(
                            Item('start_time', enabled_when='state == "idle"', width=-80),
                            Item('end_time', enabled_when='state == "idle"', width=-80),
                            Item('time_step', enabled_when='state == "idle"', width=-80),
                            Item('rseed', enabled_when='state == "idle"', width=-80),
                            Item('K', enabled_when='state == "idle"', width=-80),
                            Item('n_envelop', enabled_when='state == "idle"', width=-80),
                            label='Sweep',
                            show_border=True,
                        ),
                    ),
                    VGroup(
                        Tabbed(
                            VGroup(#Item('cursorPosition', style='readonly'),
                                Item('line_plot', editor=ComponentEditor(), show_label=False, width=500, height=400, resizable=True),
                                label='LINE',
                            ),
                            VGroup(
                                Item('matrix_plot', editor=ComponentEditor(), show_label=False, width=500, height=400, resizable=True),
                                label='COUNT',
                            ),
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
                        # Item('dual', enabled_when='state != "run"'),
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
                Action(action='save_pulse_plot', name='Save Pul Plot (.png)'),
                Action(action='save_all', name='Save All(.png) + .pys'),
                Action(action='_on_close', name='Quit'),
                name='File'
            ),
        ),
        title='Tomography (AWG)',
        buttons=[],
        resizable=True,
        width =-900,
        # height=-800,
        handler=PulsedToolHandler
    )