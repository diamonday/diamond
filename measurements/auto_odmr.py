import numpy as np

from traits.api       import SingletonHasTraits, Instance, Range, Bool, Float, Array, Str, Enum, Button, on_trait_change, Trait
from traitsui.api     import View, Item, Group, HGroup, VGroup, VSplit, Tabbed, EnumEditor, Action, Menu, MenuBar, TextEditor
from enable.api       import ComponentEditor
from chaco.api        import HPlotContainer, Plot, PlotAxis, CMapImagePlot, ColorBar, LinearMapper, ArrayPlotData, Spectral, jet

# date and time tick marks
from chaco.scales.api import CalendarScaleSystem
from chaco.scales_tick_generator import ScalesTickGenerator

import threading
import time
import logging

from tools.emod import ManagedJob
from tools.cron import CronDaemon, CronEvent

from tools.utility import GetSetItemsHandler, GetSetItemsMixin, StoppableThread, warning

from hardware.api import Scanner
import hardware.api as ha
from measurements.odmr_t import ODMR

scanner = Scanner()

class AutoODMRHandler( GetSetItemsHandler ):
    """Provides target menu."""
    
    def remove_all_targets(self,info):
        info.object.remove_all_targets()

    def forget_drift(self,info):
        info.object.forget_drift()


class AutoODMR( ManagedJob, GetSetItemsMixin ):

    # overwrite default priority from ManagedJob (default 0)
    priority = 8

    odmr = Instance(ODMR)

    power_p = Range(low= -100., high=20., value= -20, desc='Power Pmode [dBm]', label='Power Pmode[dBm]', mode='text', auto_set=False, enter_set=True)
    frequency_size = Range(low=1, high=6.4e9, value=1.0e6, desc='Start Frequency Pmode[Hz]', label='Begin Pmode[Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    frequency_delta = Range(low=1e-3, high=3.3e9, value=1.0e4, desc='frequency step Pmode[Hz]', label='Delta Pmode[Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    t_pi = Range(low=1., high=100000., value=2293., desc='length of pi pulse [ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    seconds_per_point = Range(low=10e-3, high=1, value=20e-3, desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)
    stop_time = Range(low=1., value=20, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    n_lines = Range (low=1, high=10000, value=50, desc='Number of lines in Matrix', label='Matrix lines', mode='text', auto_set=False, enter_set=True)
    
    targets         = Instance( {}.__class__, factory={}.__class__ ) # Dict traits are no good for pickling, therefore we have to do it with an ordinary dictionary and take care about the notification manually 
    target_list     = Instance( list, factory=list, args=([None],) ) # list of targets that are selectable in current_target editor
    current_target  = Enum(values='target_list')

    fit_frequency = Float(value=np.array((np.nan,)), label='frequency [Hz]')
    fit_line_width = Float(value=np.array((np.nan,)), label='line_width [Hz]') 
    fit_contrast = Float(value=np.array((np.nan,)), label='contrast [%]')

    counts = Array()
    frequency = Array()

    drift               = Array(value=np.array([0,]))
    drift_time          = Array(value=np.array([0,]))
    current_drift       = Float(value=0.)

    focus_interval = Range(low=1, high=6000, value=10, desc='Time interval between automatic focus events', label='Interval [m]', auto_set=False, enter_set=True)
    periodic_focus    = Bool(False, label='Periodic focusing')

    target_name = Str(label='name', desc='name to use when adding or removing targets')
    add_target_button       = Button(label='Add Target', desc='add target with given name')
    remove_current_target_button    = Button(label='Remove Current', desc='remove current target')
    next_target_button      = Button(label='Next Target', desc='switch to next available target')
    undo_button       = Button(label='undo', desc='undo the movement of the stage')
    
    previous_state = Instance( () )
    
    plot_data_line  = Instance( ArrayPlotData )
    plot_data_drift = Instance( ArrayPlotData )

    figure_line     = Instance( Plot, editor=ComponentEditor() )
    figure_drift    = Instance( Plot, editor=ComponentEditor() )

    def __init__(self, odmr):
        super(AutoODMR, self).__init__()
        self.odmr = odmr
        self.counts = self.odmr.counts
        self.frequency = self.odmr.frequency
        self.on_trait_change(self.update_plot_line_value, 'counts', dispatch='ui')
        self.on_trait_change(self.update_plot_line_index, 'frequency', dispatch='ui')
        self.on_trait_change(self.update_plot_drift_value, 'drift', dispatch='ui')
        self.on_trait_change(self.update_plot_drift_index, 'drift_time', dispatch='ui')
    
    @on_trait_change('next_target_button')
    def next_target(self):
        """Convenience method to switch to the next available target."""
        keys = self.targets.keys()
        key = self.current_target
        if len(keys) == 0:
            logging.getLogger().info('No target available. Add a target and try again!')
        elif not key in keys:
            self.current_target = keys[0]
        else:
            self.current_target = keys[(keys.index(self.current_target)+1)%len(keys)]

    def _targets_changed(self, name, old, new):
        l = new.keys() + [None]      # rebuild target_list for Enum trait
        l.sort()
        self.target_list = l

    def _periodic_focus_changed(self, new):
        if not new and hasattr(self, 'cron_event'):
            CronDaemon().remove(self.cron_event)
        if new:
            self.cron_event = CronEvent(self.submit, min=range(0,60,self.focus_interval))
            CronDaemon().register(self.cron_event)


    def add_target(self, key, f=None):
        if f is None:
            odmr = self.odmr
            f = np.array([odmr.fit_frequencies[0],])
        if self.targets == {}:
            self.forget_drift()
        if self.targets.has_key(key):
            if warning('A target with this name already exists.\nOverwriting will move all targets.\nDo you want to continue?'):
                self.current_drift = f - self.targets[key]
                self.forget_drift()
            else:
                return
        else:
            f = f - self.current_drift
            self.targets[key] = f
        self.trait_property_changed('targets', self.targets)    # trigger event such that Enum is updated and Labels are redrawn
        self.confocal.show_labels=True

    def remove_target(self, key):
        if not key in self.targets:
            logging.getLogger().info('Target cannot be removed. Target does not exist.')
            return
        self.targets.pop(key)        # remove target from dictionary
        self.trait_property_changed('targets', self.targets)    # trigger event such that Enum is updated and Labels are redrawn
        
    def remove_all_targets(self):
        self.targets = {}

    def forget_drift(self):
        targets = self.targets
        # reset coordinates of all targets according to current drift
        for key in targets:
            targets[key] += self.current_drift
        # trigger event such that target labels are redrawn
        self.trait_property_changed('targets', self.targets)
        # set current_drift to 0 and clear plot
        self.current_drift = 0.
        self.drift_time = np.array([time.time(),])
        self.drift = np.array([0,])
        
    def _add_target_button_fired(self):
        self.add_target( self.target_name )
        
    def _remove_current_target_button_fired(self):
        self.remove_target( self.current_target )

    def _run(self):
        
        logging.getLogger().debug("trying run.")
        
        try:
            self.state='run'
            ha.PulseGenerator().Light()

            if self.current_target is None:
                self.focus()
            else: # focus target
                current_f = self.targets[self.current_target]
                self.focus()
                current_f = self.fit_frequency
                self.current_drift = self.fit_frequency - current_f
                self.drift = np.append(self.drift, (self.current_drift,), axis=0)
                self.drift_time = np.append(self.drift_time, time.time())
                logging.getLogger().debug('Drift: %.2f'%self.current_drift)
        finally:
            self.state = 'idle'

    def focus(self):
            """
            Focuses around current frequency in x, y, and z-direction.
            """
            fp = self.odmr.fit_frequencies[0]
            ##+scanner.getXRange()[1]
            f_begin = fp - self.frequency_size
            f_end= fp + self.frequency_size

            self.odmr.number_of_resonances = 1
            self.odmr.frequency_begin_p = f_begin
            self.odmr.frequency_end_p = f_end
            self.odmr.frequency_delta_p = self.frequency_delta
            self.odmr.stop_time = self.stop_time
            
            odmr_priority = self.odmr.priority
            self.odmr.priority = 9

            self.odmr.submit()
            from tools.emod import JobManager
            print(JobManager().queue,JobManager().running)
            time.sleep(2)
            print(JobManager().queue,JobManager().running)
            self.odmr.priority = odmr_priority

            if (not np.isnan(self.odmr.fit_frequencies[0])) and (not np.isnan(self.odmr.fit_line_width)) and (not np.isnan(self.odmr.fit_contrast)):
                print(self.odmr.fit_frequencies[0],self.odmr.state)
                self.fit_frequency = self.odmr.fit_frequencies[0]
                self.fit_line_width = self.odmr.fit_line_width[0]
                self.fit_contrast = self.odmr.fit_contrast[0]
                logging.getLogger().info('Auto ODMR: %.2f, %.2f, %.2f' %(self.fit_frequency, self.fit_line_width, self.fit_contrast))
            else:
                logging.getLogger().info('Auto ODMR: Fitting failed!')
    
    def _plot_data_line_default(self):
        line_data = ArrayPlotData(frequency=np.array((0., 1.)), counts=np.array((0., 0.)), fit=np.array((0., 0.))) 
        return line_data

    def _plot_data_drift_default(self):
        return ArrayPlotData(time=self.drift_time, freq_drift=self.drift)
    
    def _figure_line_default(self):
        line_plot = Plot(self.plot_data_line, width=100, height=100, padding=8, padding_left=64, padding_bottom=32)
        line_plot.plot(('frequency', 'counts'), style='line', color='blue')
        line_plot.index_axis.title = 'Frequency [MHz]'
        line_plot.value_axis.title = 'Fluorescence [ counts / s ]'
        return line_plot

    def _figure_drift_default(self):
        plot = Plot(self.plot_data_drift, width=100, height=100, padding=8, padding_left=64, padding_bottom=32)
        plot.plot(('time','freq_drift'), type='line', color='black')
        bottom_axis = PlotAxis(
            plot,
            orientation="bottom",
            tick_generator=ScalesTickGenerator(scale=CalendarScaleSystem())
        )
        plot.index_axis=bottom_axis
        plot.index_axis.title = 'time'
        plot.value_axis.title = 'Frequency drift [MHz]'
        plot.legend.visible=True
        return plot        

    def update_plot_line_value(self):
        self.plot_data_line.set_data('counts', self.odmr.counts)

    def update_plot_line_index(self):
        self.plot_data_line.set_data('frequency', self.odmr.frequency * 1e-6)

    def update_plot_drift_value(self):
        if len(self.drift) == 1:
            self.plot_data_drift.set_data('freq_drift', np.array(()))         
        else:
            self.plot_data_drift.set_data('freq_drift', self.drift)

    def update_plot_drift_index(self):
        if len(self.drift_time) == 0:
            self.plot_data_drift.set_data('time', np.array(()))
        else:
            self.plot_data_drift.set_data('time', self.drift_time)

    traits_view = View(
        VGroup(
            HGroup(
                Item('submit_button', show_label=False),
                Item('remove_button', show_label=False),
                Item('priority'),
                Item('state', style='readonly'),
                Item('undo_button', show_label=False),
            ),
            Group(
                VGroup(
                    HGroup(
                        Item('target_name'),
                        Item('add_target_button', show_label=False),
                    ),
                    HGroup(
                        Item('current_target'),
                        Item('next_target_button', show_label=False),
                        Item('remove_current_target_button', show_label=False),
                    ),
                    HGroup(
                        Item('periodic_focus'),
                        Item('focus_interval', enabled_when='not periodic_focus'),
                    ),
                    label='tracking',
                ),
                VGroup(
                    HGroup(
                        Item('frequency_size'),
                        Item('frequency_delta'),
                    ),
                    HGroup(
                        Item('seconds_per_point'),
                        Item('power_p'),
                    ),
                    HGroup(
                        Item('stop_time'),
                        Item('t_pi'),
                    ),                    
                    label='Settings',
                ),
                layout='tabbed'
            ),
            VSplit(
                Item('figure_line', show_label=False, resizable=True),
                Item('figure_drift', show_label=False, resizable=True),
            ),
        ),
        menubar = MenuBar(
            Menu(
                Action(action='save', name='Save (.pyd or .pys)'),
                Action(action='load', name='Load'),
                Action(action='_on_close', name='Quit'),
                name='File'
            ),
            Menu(
                Action(action='remove_all_targets', name='Remove All'),
                Action(action='forget_drift', name='Forget Drift'),
                name='Target'
            )
        ),
        title='Auto ODMR', width=500, height=700, buttons=[], resizable=True,
        handler=AutoODMRHandler
    )

    get_set_items=['odmr','targets','current_target','current_drift','drift','drift_time','periodic_focus',
                   'power_p','t_pi','frequency_size','frequency_delta','stop_time','focus_interval' ]
    get_set_order=['odmr','targets']

    
# testing

if __name__ == '__main__':
    
    logging.getLogger().setLevel(logging.DEBUG)
    
    from emod import JobManager
    
    JobManager().start()

    from cron import CronDaemon
    
    CronDaemon().start()
    
    c = Confocal()
    c.edit_traits()
    a = AutoFocus(c)
    a.edit_traits()