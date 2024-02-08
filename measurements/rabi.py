import numpy as np

from traits.api import Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group, Label

import logging
import time

import hardware.api as ha

from chaco.api import ArrayPlotData, Plot, Spectral, PlotLabel

from pulsed import Pulsed
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

class Rabi( Pulsed ):
    
    """Defines a Rabi measurement."""

    #switch      = Enum( 'mw_x','mw_b','mw_c',   desc='switch to use for different microwave source',     label='switch' )
    switch      = Enum( 'mw_x','mw_y',   desc='switch to use for different microwave source',     label='switch' )
    frequency   = Range(low=1,      high=20e9,  value=2.879837e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.6e' %x))
    power       = Range(low=-100.,  high=25.,   value=-16,      desc='microwave power',     label='power [dBm]',    mode='text', auto_set=False, enter_set=True)

    tau_begin   = Range(low=0., high=1e8,       value=1.5,     desc='tau begin [ns]',  label='tau begin [ns]',   mode='text', auto_set=False, enter_set=True)
    tau_end     = Range(low=1., high=1e8,       value=4000.,     desc='tau end [ns]',    label='tau end [ns]',     mode='text', auto_set=False, enter_set=True)
    tau_delta   = Range(low=1., high=1e6,       value=30.,      desc='delta tau [ns]',  label='delta tau [ns]',   mode='text', auto_set=False, enter_set=True)
    laser       = Range(low=1., high=100000.,   value=3000.,    desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    wait        = Range(low=0., high=100000.,   value=1000.,    desc='wait [ns]',       label='wait [ns]',        mode='text', auto_set=False, enter_set=True)

    tau = Array( value=np.array((0.,1.)) )

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)
        
    def start_up(self):
        ha.PulseGenerator().Night()
        ha.MicrowaveA().setOutput(self.power, self.frequency)
        '''
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveE().setOutput(self.power, self.frequency)
        elif self.switch=='mw_c':
            ha.MicrowaveD().setOutput(self.power, self.frequency)
        '''
            
    def shut_down(self):
        ha.PulseGenerator().Light()
        ha.MicrowaveA().setOutput(None, self.frequency)
        """
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveD().setOutput(None, self.frequency)
        elif self.switch=='mw_c':
            ha.MicrowaveD().setOutput(None, self.frequency)
        """


    def generate_sequence(self):
        MW = self.switch
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        sequence += [ ([  ], 200),(['sequence'], 100  )  ]
        sequence += [ ([  ], 1000 )  ]
        for t in tau:
            #sequence += [  ([MW, 'mw'],t),  (['laser','aom'],laser),  ([],wait)  ]
            #sequence += [  ([MW],t),  (['laser','aom'],laser),  ([],wait)  ]
            sequence += [  (['A','B',MW],t),  (['B','laser','aom'],laser),  (['B',],wait)  ]
        return sequence

    get_set_items = Pulsed.get_set_items + ['frequency','power','switch','tau_begin','tau_end','tau_delta','laser','wait','tau']
    get_set_order = ['tau','time_bins','count_data']

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly',format_str='%.f'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('switch', style='custom'),
                                                   Item('frequency',     width=-80, enabled_when='state == "idle"'),
                                                   Item('power',         width=-80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('tau_begin',     width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_end',       width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_delta',     width=-80, enabled_when='state == "idle"'),
                                                   Item('stop_time'),),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser',         width=-80, enabled_when='state == "idle"'),
                                                   Item('wait',          width=-80, enabled_when='state == "idle"'),
                                                   Item('record_length', width=-80, enabled_when='state == "idle"'),
                                                   Item('bin_width',     width=-80, enabled_when='state == "idle"'),),
                                     label='settings'),
                              ),
                        ),
                       title='Rabi Measurement',
                  )
    
class Rabi1( Pulsed ):
    
    """Defines a Rabi measurement."""

    switch      = Enum( 'mw_x','mw_b','mw_y',   desc='switch to use for different microwave source',     label='switch' )
    frequency   = Range(low=1,      high=20e9,  value=2.8705e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power       = Range(low=-100.,  high=25.,   value=-24,      desc='microwave power',     label='power [dBm]',    mode='text', auto_set=False, enter_set=True)

    tau_begin   = Range(low=0., high=1e8,       value=1.5,     desc='tau begin [ns]',  label='tau begin [ns]',   mode='text', auto_set=False, enter_set=True)
    tau_end     = Range(low=1., high=1e8,       value=5000.,     desc='tau end [ns]',    label='tau end [ns]',     mode='text', auto_set=False, enter_set=True)
    tau_delta   = Range(low=1., high=1e6,       value=30.,      desc='delta tau [ns]',  label='delta tau [ns]',   mode='text', auto_set=False, enter_set=True)
    laser       = Range(low=1., high=100000.,   value=3000.,    desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    wait        = Range(low=0., high=100000.,   value=1000.,    desc='wait [ns]',       label='wait [ns]',        mode='text', auto_set=False, enter_set=True)

    tau = Array( value=np.array((0.,1.)) )

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)
        
    def start_up(self):
        ha.PulseGenerator().Night()
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveD().setOutput(self.power, self.frequency)
        elif self.switch=='mw_f':
            ha.MicrowaveF().setOutput(self.power, self.frequency)
            
    def shut_down(self):
        ha.PulseGenerator().Light()
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveD().setOutput(None, self.frequency)
        elif self.switch=='mw_f':
            ha.MicrowaveF().setOutput(None, self.frequency)


    def generate_sequence(self):
        MW = self.switch
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        for t in tau:
            #sequence += [  ([MW,'mw'],t),  (['laser','aom'],laser),  ([],wait)  ]
            sequence += [  ([MW],t),  (['laser','aom'],laser),  ([],wait)  ]
        sequence += [ (['sequence'], 100  )  ]
        return sequence

    get_set_items = Pulsed.get_set_items + ['frequency','power','switch','tau_begin','tau_end','tau_delta','laser','wait','tau']
    get_set_order = ['tau','time_bins','count_data']

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly',format_str='%.f'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('switch', style='custom'),
                                                   Item('frequency',     width=-80, enabled_when='state == "idle"'),
                                                   Item('power',         width=-80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('tau_begin',     width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_end',       width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_delta',     width=-80, enabled_when='state == "idle"'),
                                                   Item('stop_time'),),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser',         width=-80, enabled_when='state == "idle"'),
                                                   Item('wait',          width=-80, enabled_when='state == "idle"'),
                                                   Item('record_length', width=-80, enabled_when='state == "idle"'),
                                                   Item('bin_width',     width=-80, enabled_when='state == "idle"'),),
                                     label='settings'),
                              ),
                        ),
                       title='Rabi Measurement',
                  )

''' 
class Rabi_fit( Pulsed, ManagedJob, GetSetItemsMixin ):
    
    """Defines a Rabi measurement."""

    switch      = Enum( 'mw_x','mw_b','mw_c',   desc='switch to use for different microwave source',     label='switch' )
    frequency   = Range(low=1,      high=20e9,  value=2.879837e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power       = Range(low=-100.,  high=25.,   value=-16,      desc='microwave power',     label='power [dBm]',    mode='text', auto_set=False, enter_set=True)

    tau_begin   = Range(low=0., high=1e8,       value=1.5,     desc='tau begin [ns]',  label='tau begin [ns]',   mode='text', auto_set=False, enter_set=True)
    tau_end     = Range(low=1., high=1e8,       value=4000.,     desc='tau end [ns]',    label='tau end [ns]',     mode='text', auto_set=False, enter_set=True)
    tau_delta   = Range(low=1., high=1e6,       value=30.,      desc='delta tau [ns]',  label='delta tau [ns]',   mode='text', auto_set=False, enter_set=True)
    laser       = Range(low=1., high=100000.,   value=3000.,    desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    wait        = Range(low=0., high=100000.,   value=1000.,    desc='wait [ns]',       label='wait [ns]',        mode='text', auto_set=False, enter_set=True)

    tau = Array( value=np.array((0.,1.)) )
    
    switch      = Enum( 'mw_x','mw_b','mw_c',   desc='switch to use for different microwave source',     label='switch' )
    frequency   = Range(low=1,      high=20e9,  value=2.879837e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power       = Range(low=-100.,  high=25.,   value=-16,      desc='microwave power',     label='power [dBm]',    mode='text', auto_set=False, enter_set=True)

    tau_begin   = Range(low=0., high=1e8,       value=1.5,     desc='tau begin [ns]',  label='tau begin [ns]',   mode='text', auto_set=False, enter_set=True)
    tau_end     = Range(low=1., high=1e8,       value=4000.,     desc='tau end [ns]',    label='tau end [ns]',     mode='text', auto_set=False, enter_set=True)
    tau_delta   = Range(low=1., high=1e6,       value=30.,      desc='delta tau [ns]',  label='delta tau [ns]',   mode='text', auto_set=False, enter_set=True)
    laser       = Range(low=1., high=100000.,   value=3000.,    desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    wait        = Range(low=0., high=100000.,   value=1000.,    desc='wait [ns]',       label='wait [ns]',        mode='text', auto_set=False, enter_set=True)

    tau = Array( value=np.array((0.,1.)) )
    
    get_set_items = Pulsed.get_set_items + ['frequency','power','switch','tau_begin','tau_end','tau_delta','laser','wait']
    
    #fit parameters
    fit_parameters = np.ones(4)
    rabi_period    = Float(value=0.0, label='T')
    rabi_frequency = Float(value=0.0, label='f')
    rabi_contrast  = Float(value=0.0, label='I')
    rabi_offset    = Float(value=0.0, label='t_0')
    pi2            = Float(value=0.0, label='pi/2')
    pi             = Float(value=0.0, label='pi')
    pi32           = Float(value=0.0, label='3pi/2')
    
    get_set_items += ['freq1', 'amp1', 'show_fit', 'fit_parameters', 'rabi_frequency', 'rabi_period',
                                            'rabi_offset', 'rabi_contrast', 'pi2', 'pi', 'pi32']
    
    line_plot = Instance(Plot, editor=ComponentEditor())

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)
        
    def start_up(self):
        ha.PulseGenerator().Night()
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveE().setOutput(self.power, self.frequency)
        elif self.switch=='mw_c':
            ha.MicrowaveD().setOutput(self.power, self.frequency)
            
    def shut_down(self):
        ha.PulseGenerator().Light()
        """
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveD().setOutput(None, self.frequency)
        elif self.switch=='mw_c':
            ha.MicrowaveD().setOutput(None, self.frequency)
        """


    def generate_sequence(self):
        MW = self.switch
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        sequence += [ ([  ], 200),(['sequence'], 100  )  ]
        sequence += [ ([  ], 1000 )  ]
        for t in tau:
            #sequence += [  ([MW, 'mw'],t),  (['laser','aom'],laser),  ([],wait)  ]
            sequence += [  ([MW],t),  (['laser','aom'],laser),  ([],wait)  ]
        return sequence

    get_set_items += ['tau']
    get_set_order = ['tau','time_bins','count_data']

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly',format_str='%.f'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('switch', style='custom'),
                                                   Item('frequency',     width=-80, enabled_when='state == "idle"'),
                                                   Item('power',         width=-80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('tau_begin',     width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_end',       width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_delta',     width=-80, enabled_when='state == "idle"'),
                                                   Item('stop_time'),),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser',         width=-80, enabled_when='state == "idle"'),
                                                   Item('wait',          width=-80, enabled_when='state == "idle"'),
                                                   Item('record_length', width=-80, enabled_when='state == "idle"'),
                                                   Item('bin_width',     width=-80, enabled_when='state == "idle"'),),
                                     label='settings'),
                              ),
                        ),
                       title='Rabi Measurement',
                  )       
                  
    # plotting
    def _create_line_plot(self):
        line_data = ArrayPlotData(frequency=np.array((0., 1.)), counts=np.array((0., 0.)), fit=np.array((0., 0.))) 
        line_plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=32)
        line_plot.plot(('frequency', 'counts'), style='line', color='blue')
        line_plot.index_axis.title = 'Frequency [MHz]'
        line_plot.value_axis.title = 'Fluorescence counts'
        line_label = PlotLabel(text='', hjustify='left', vjustify='bottom', position=[64, 128])
        line_plot.overlays.append(line_label)
        self.line_label = line_label
        self.line_data = line_data
        self.line_plot = line_plot
        
    def _update_line_data_index(self):
        self.line_data.set_data('frequency', self.frequency * 1e-6)
        self.counts_matrix = self._counts_matrix_default()

    def _update_line_data_value(self):
        self.line_data.set_data('counts', self.counts)

    def _update_line_data_fit(self):
        if not np.isnan(self.fit_parameters[0]):            
            self.line_data.set_data('fit', fitting.NLorentzians(*self.fit_parameters)(self.frequency))
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
'''



class Rabi_2smiq( Pulsed ):
    
    """Defines a Rabi measurement."""

    switch      = Enum( 'mw_x','mw_2','mw_y',   desc='switch to use for different microwave source',     label='switch' )
    frequency   = Range(low=1,      high=20e9,  value=2.8705e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power       = Range(low=-100.,  high=25.,   value=-24,      desc='microwave power',     label='power [dBm]',    mode='text', auto_set=False, enter_set=True)

    tau_begin   = Range(low=0., high=1e8,       value=1.5,     desc='tau begin [ns]',  label='tau begin [ns]',   mode='text', auto_set=False, enter_set=True)
    tau_end     = Range(low=1., high=1e8,       value=5000.,     desc='tau end [ns]',    label='tau end [ns]',     mode='text', auto_set=False, enter_set=True)
    tau_delta   = Range(low=1., high=1e6,       value=30.,      desc='delta tau [ns]',  label='delta tau [ns]',   mode='text', auto_set=False, enter_set=True)
    laser       = Range(low=1., high=100000.,   value=3000.,    desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    wait        = Range(low=0., high=100000.,   value=1000.,    desc='wait [ns]',       label='wait [ns]',        mode='text', auto_set=False, enter_set=True)

    tau = Array( value=np.array((0.,1.)) )

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)
        
    def start_up(self):
        ha.PulseGenerator().Night()
        if self.switch=='mw_2':
            ha.MicrowaveB().setOutput(self.power, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
            
    def shut_down(self):
        ha.PulseGenerator().Light()
        if self.switch=='mw_2':
            ha.MicrowaveB().setOutput(None, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(None, self.frequency)


    def generate_sequence(self):
        MW = self.switch
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        if MW =='mw_2':
            trig = ['B']
        elif MW =='mw_x' or MW =='mw_y':
            trig = ['A', MW]
        for t in tau:
            #sequence += [  ([MW,'mw'],t),  (['laser','aom'],laser),  ([],wait)  ]
            sequence += [  (trig,t),  (['laser','aom'],laser),  ([],wait)  ]
        sequence += [ (['sequence'], 100  )  ]
        return sequence

    get_set_items = Pulsed.get_set_items + ['frequency','power','switch','tau_begin','tau_end','tau_delta','laser','wait','tau']
    get_set_order = ['tau','time_bins','count_data']

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly',format_str='%.f'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('switch', style='custom'),
                                                   Item('frequency',     width=-80, enabled_when='state == "idle"'),
                                                   Item('power',         width=-80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('tau_begin',     width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_end',       width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_delta',     width=-80, enabled_when='state == "idle"'),
                                                   Item('stop_time'),),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser',         width=-80, enabled_when='state == "idle"'),
                                                   Item('wait',          width=-80, enabled_when='state == "idle"'),
                                                   Item('record_length', width=-80, enabled_when='state == "idle"'),
                                                   Item('bin_width',     width=-80, enabled_when='state == "idle"'),),
                                     label='settings'),
                              ),
                        ),
                       title='Rabi Measurement with 2 smiq',
                  )

class Rabi_3smiq( Pulsed ):
    
    """Defines a Rabi measurement."""

    switch      = Enum( 'mw_13','mw_24','mw_all_x','mw_all_y',   desc='switch to use for different microwave source',     label='switch' )
    frequency   = Range(low=1,      high=20e9,  value=2.8705e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power       = Range(low=-100.,  high=25.,   value=-24,      desc='microwave power',     label='power [dBm]',    mode='text', auto_set=False, enter_set=True)

    tau_begin   = Range(low=0., high=1e8,       value=1.5,     desc='tau begin [ns]',  label='tau begin [ns]',   mode='text', auto_set=False, enter_set=True)
    tau_end     = Range(low=1., high=1e8,       value=5000.,     desc='tau end [ns]',    label='tau end [ns]',     mode='text', auto_set=False, enter_set=True)
    tau_delta   = Range(low=1., high=1e6,       value=30.,      desc='delta tau [ns]',  label='delta tau [ns]',   mode='text', auto_set=False, enter_set=True)
    laser       = Range(low=1., high=100000.,   value=3000.,    desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    wait        = Range(low=0., high=100000.,   value=1000.,    desc='wait [ns]',       label='wait [ns]',        mode='text', auto_set=False, enter_set=True)

    tau = Array( value=np.array((0.,1.)) )

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)
        
    def start_up(self):
        ha.PulseGenerator().Night()
        if self.switch=='mw_13':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_24':
            ha.MicrowaveB().setOutput(self.power, self.frequency)
        elif self.switch=='mw_all_x' or self.switch=='mw_all_y':
            ha.MicrowaveC().setOutput(self.power, self.frequency)
            
    def shut_down(self):
        ha.PulseGenerator().Light()
        if self.switch=='mw_13':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_24':
            ha.MicrowaveB().setOutput(None, self.frequency)
        elif self.switch=='mw_all_x' or self.switch=='mw_all_y':
            ha.MicrowaveC().setOutput(None, self.frequency)


    def generate_sequence(self):
        MW = self.switch
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        trig = [MW]
        for t in tau:
            sequence += [  (trig,t),  (['laser','aom'],laser),  ([],wait)  ]
        sequence += [ (['sequence'], 100  )  ]
        return sequence

    get_set_items = Pulsed.get_set_items + ['frequency','power','switch','tau_begin','tau_end','tau_delta','laser','wait','tau']
    get_set_order = ['tau','time_bins','count_data']

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly',format_str='%.f'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('switch', style='custom'),
                                                   Item('frequency',     width=-80, enabled_when='state == "idle"'),
                                                   Item('power',         width=-80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('tau_begin',     width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_end',       width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_delta',     width=-80, enabled_when='state == "idle"'),
                                                   Item('stop_time'),),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser',         width=-80, enabled_when='state == "idle"'),
                                                   Item('wait',          width=-80, enabled_when='state == "idle"'),
                                                   Item('record_length', width=-80, enabled_when='state == "idle"'),
                                                   Item('bin_width',     width=-80, enabled_when='state == "idle"'),),
                                     label='settings'),
                              ),
                        ),
                       title='Rabi Measurement with 3 smiq',
                  )
