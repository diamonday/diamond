import numpy as np

fft = np.fft.fft
ifft = np.fft.ifft

from hardware.waveform import *

from traits.api import Instance, Property, Range, String, Float, Int, Bool, Array, List, Enum, Trait,\
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
    amp1  = Range(low=0.0, high=1.0, value=1.0, desc='Waveform amplitude factor', label='wfm amp', auto_set=False, enter_set=True)
    main_dir = String('\Tomo', desc='.WFM/.PAT folder in AWG', label='Waveform Dir.')
    offset = Range(low=-20e6, high=20e6, value=0.0, desc='Offset frequency', label='Offset frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    freq_awg = Range(low=1, high=1e9, value=40e6, desc='AWG frequency', label='AWG frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    freq_tot = Range(low=1, high=20e9, value=2.87e9, desc='Total frequency', label='Total frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    detuning = Range(low=-50e6, high=50e6, value=1.0e6, desc='AWG frequency', label='Detuning [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    phase_sin = Range(low=-100.0, high=100.0, value=0.0, desc='Multiple of pi', label='Phase', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%.2f'))
    
    on_AWG = Bool(False, label='Detuning on AWG')

    t_pi = Range(low=0.0, high=2000, value=200., desc='duration of pi pulse', label='pi duration', auto_set=False, enter_set=True)
    t_x = Range(low=0.0, high=2000, value=0., desc='duration of x pulse', label='x duration', auto_set=False, enter_set=True)
    t_y = Range(low=0.0, high=2000, value=0., desc='duration of y pulse', label='y duration', auto_set=False, enter_set=True)
    add_ref0 = Bool(False, label='Add REF0', desc='Whether to add Ref0 Waveform')

    get_set_items = Pulsed.get_set_items + [
        'freq_tot', 'freq_awg', 'detuning', 'offset', 'amp1', 'on_AWG',
        't_pi', 't_x', 't_y', 'phase_sin', 'add_ref0',
    ]

    #parameters for f(x)
    K = Range(low=0, high=100, value=10, desc='Number of Fourier components', label='K', mode='text', auto_set=False, enter_set=True)
    n_envelop = Range(low=0, high=100, value=2, desc='Dimension of the system', label='number of envelop', mode='text', auto_set=False, enter_set=True)
    rseed = Int(low=0, high=1000, value=0, desc='Random Seed', label='Seed', mode='text',auto_set=False, enter_set=True)
    f0 = Range(low=0.0, high=10e6, value=4.0e6, desc='Random puluse frequency', label='f0 [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))

    amp_f = Array()
    freq_f = Array()
    phase_f = Array()
    amp_DC = Array()
    
    DC = Bool(False, label='DC', desc='Output DC wave for calibration')
    amp_start  = Range(low=0.0, high=1.0, value=0.0, desc='Lowest amplitude', label='wfm start', auto_set=False, enter_set=True)
    amp_end  = Range(low=0.0, high=1.0, value=1.0, desc='Highest amplitude', label='wfm end', auto_set=False, enter_set=True)

    get_set_items += [
        'K', 'n_envelop', 'rseed','f0',
        'amp_f','freq_f','phase_f',
        'DC','amp_start','amp_end','amp_DC',
    ]
    
    def __init__(self, pulse_generator, time_tagger, microwave, awg, **kwargs):
        self.dual = False
        super(Tomo, self).__init__(pulse_generator, time_tagger, microwave, awg, **kwargs)
        self.freq_tot = self.freq + self.freq_awg + self.offset + self.detuning
        
        r = int(self.rseed)
        K = int(self.K)

        np.random.seed(r)
        self.amp_f = np.random.rand(self.n_envelop,K)
        self.amp_f /= self.amp_f.sum(axis=1)[:,None]  # Normalization
        self.freq_f = np.random.rand(self.n_envelop,K)
        self.phase_f = np.random.rand(self.n_envelop,K)
        self.amp_DC = np.linspace(self.amp_start, self.amp_end, self.n_envelop)
    
    # Submit
    def start_up(self):
        """Override for additional stuff to be executed at start up."""
        self.pulse_generator.Night()
        if self.on_AWG:
            self.microwave.setOutput(self.power, self.freq + self.offset)
        else:
            self.microwave.setOutput(self.power, self.freq + self.offset + self.detuning)
        self.awg.set_output(0b11)
        self.awg.run()
    
    def shut_down(self):
        """Override for additional stuff to be executed at shut down."""
        self.pulse_generator.Light()
        if self.on_AWG:
            self.microwave.setOutput(None, self.freq + self.offset)
        else:
            self.microwave.setOutput(None, self.freq + self.offset + self.detuning)
        self.awg.stop()
        self.awg.set_output(0b00)

    # Sequence ############################
    def generate_sequence(self):
        tau   = self.tau
        laser = self.laser
        wait  = self.wait
        t_pi  = self.t_pi
        t_x   = self.t_x
        t_y   = self.t_y
        trig_interval = self.trig_interval
        trig_delay = self.trig_delay

        sequence = [
            (['aom'], laser),
            ([], wait)
        ]

        for i in range(self.n_envelop):
            for t in tau:
                # Fill zero to match the duty cycle of laser
                sequence.append(([],tau.max() - t))
                sequence.append((['awg'], trig_interval))
                sequence.append(([], trig_delay + t + t_x + t_y))
                # Add switch at the end
                #sequence.append((['mw_y'], trig_delay + t + t_x + t_y))
                sequence.append((['laser', 'green'], laser))               
                sequence.append(([], wait))
        
        # Bright
        if self.add_ref0:
            # Fill zero to match the duty cycle of laser
            sequence.append(([],tau.max() - t_pi))
            sequence.append((['awg'], trig_interval))       #idle
            sequence.append(([], trig_delay + t_pi))
            sequence.append((['laser', 'green'], laser))    #0 state readout
            sequence.append(([], wait))
        else:
            # Fill zero to match the duty cycle of laser
            sequence.append(([],tau.max()))
            sequence.append((['laser', 'green'], laser))    #0 state readout
            sequence.append(([], wait))

        # Dark
        # Fill zero to match the duty cycle of laser
        sequence.append(([],tau.max()))
        sequence.append((['awg'], trig_interval))           #reference pi-pulse
        sequence.append(([], trig_delay + t_pi))
        sequence.append((['laser', 'green'], laser))        #-1 state readout
        sequence.append(([], wait))

        sequence.append((['sequence'], 100))

        return sequence
    
    def compile_waveforms(self):
        # awg frequency in terms of sampling rate
        # Detuned AWG frequency
        f_det = self.detuning / self.sampling
        # Resonance AWG frequency
        f_res = self.freq_awg / self.sampling
        phase = self.phase_sin*np.pi
        
        min_duration = 256 * self.sampling / 1.0e9
        print(min_duration)

        tau = self.tau * self.sampling / 1.0e9
        t_x = self.t_x * self.sampling / 1.0e9
        t_y = self.t_y * self.sampling / 1.0e9
        t_pi = self.t_pi * self.sampling / 1.0e9

        # pulse objects
        operatorx = Sin(t_x, freq=f_res, amp=self.amp1, phase=phase)
        operatory = Sin(t_y, freq=f_res, amp=self.amp1, phase=phase + np.pi/2.0)
        #sine = Sin(0, freq=f_det, amp=self.amp1, phase=phase)
        if self.on_AWG:
            sine = Sin(0, freq=f_res + f_det, amp=self.amp1, phase=phase)
        else:
            sine = Sin(0, freq=f_res, amp=self.amp1, phase=phase)
        ref = Sin(t_pi, freq=f_res, amp=self.amp1, phase=phase)
        zero = Idle(1)
        point_fill = Idle(1) # a waveform contain min. 2400 points
        
        # pulse sequence        
        
        self.waves = []
        seq=Sequence('TOMO')
        
        if self.DC:
            for p,a in enumerate(self.amp_DC):
                sine.amp = a
                pulse_sequence = [zero, sine, zero] + [point_fill]
                for i,t in enumerate(tau):
                    # increment microwave duration
                    sine.duration = t
                    if t < min_duration:
                        point_fill.duration = np.ceil(min_duration - t)
                    name = 'DC_%02i_%03i' % (p,i)
                    # file_type=0 -> .WFM
                    # file_type=1 -> .PAT
                    waves = IQWaveform(name, pulse_sequence, file_type=0)
                    self.waves.append(waves[0])
                    self.waves.append(waves[1])
                    seq.append(waves, wait=True)
                    
        else:
            for p in range(self.n_envelop):
                ffx = self.f0 * (self.freq_f[p])/self.sampling
                envelop = Envelop(duration=0, amp_li=self.amp_f[p], freq_li=ffx, phase_li=self.phase_f[p])
                mw = sine*envelop

                pulse_sequence = [zero, operatorx, operatory , mw, zero] + [point_fill]

                for i,t in enumerate(tau):
                    # increment microwave duration
                    mw.duration = t
                    if t < min_duration:
                        point_fill.duration = np.ceil(min_duration - t)
                    name = 'FX_%02i_%03i' % (p,i)
                    # file_type=0 -> .WFM
                    # file_type=1 -> .PAT
                    waves = IQWaveform(name, pulse_sequence, file_type=0)
                    self.waves.append(waves[0])
                    self.waves.append(waves[1])
                    seq.append(waves, wait=True)

        if self.add_ref0:
            point_fill.duration = min_duration
            pulse_sequencer0 = [point_fill]
            namer0 = 'REF0'
            waver0 = IQWaveform(namer0, pulse_sequencer0, file_type=0)
            self.waves.append(waver0[0])
            self.waves.append(waver0[1])
            seq.append(waver0, wait=True)
        
        if t_pi < min_duration:
            point_fill.duration = np.ceil(min_duration - t_pi)
        pulse_sequencer1 = [zero, ref, zero] + [point_fill]
        namer1 = 'REF1'
        waver1 = IQWaveform(namer1, pulse_sequencer1, file_type=0)
        
        self.waves.append(waver1[0])
        self.waves.append(waver1[1])
        seq.append(waver1, wait=True)

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
            plot.plot(('index','y%d' % i), color='auto', id='0', name='y%d' % i)
        
        plot.plot(('index', 'bright'), color='red', line_width=2, name='bright')
        plot.plot(('index', 'dark'), color='black', line_width=2, name='dark')
        
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
    
    @on_trait_change('freq, offset, freq_awg, detuning')
    def _update_freq_tot(self):
        self.freq_tot = self.freq + self.offset + self.freq_awg + self.detuning
    
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
    
    @on_trait_change('n_envelop, amp_start, amp_end')
    def _amp_value(self):
        self.amp_DC = np.linspace(self.amp_start, self.amp_end, self.n_envelop)    

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
                            Item('main_dir', style='readonly'),
                            label='AWG',
                            show_border=True,
                        ),
                        HGroup(
                            Item('freq', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e'%x), width=-80),
                            Item('offset', enabled_when='state == "idle"'),
                            Item('power', width=-40),
                            label='SMIQ',
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item('freq_tot', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.6f GHz' % (x*1e-9))),
                        Item('freq_awg', enabled_when='state == "idle"',),
                        Item('detuning', enabled_when='state == "idle"'),
                        Item('on_AWG', enabled_when='state == "idle"'),
                        label='Frequency',
                        show_border=True,
                    ),
                    HGroup(
                        Item('add_ref0', enabled_when='state == "idle"'),
                        Item('t_pi', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f'%x), width=-80),
                        Item('t_x', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f'%x), width=-80),
                        Item('t_y', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f'%x), width=-80),
                        Item('phase_sin', enabled_when='state == "idle"'),
                        Item('amp1', enabled_when='state == "idle"', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.2f'%x), width=-80),
                        label='Waveform',
                        show_border=True,
                    ),
                    HGroup(
                        Item('start_time', enabled_when='state == "idle"', width=-80),
                        Item('end_time', enabled_when='state == "idle"', width=-80),
                        Item('time_step', enabled_when='state == "idle"', width=-80),
                        label='Sweep',
                        show_border=True,
                    ),
                    HGroup(
                        Item('rseed', enabled_when='state == "idle"', width=-80),
                        Item('K', enabled_when='state == "idle"', width=-80),
                        Item('n_envelop', enabled_when='state == "idle"', width=-80),
                        Item('f0', enabled_when='state == "idle"', width=-80),
                        label='Parameters',
                        show_border=True,
                    ),
                    HGroup(
                        Item('DC', enabled_when='state == "idle"'),
                        Item('amp_start', enabled_when='state == "idle"', width=-80),
                        Item('amp_end', enabled_when='state == "idle"', width=-80),
                        label='Calibration',
                        show_border=True,
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
                            Item('trig_delay', width=-80),
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