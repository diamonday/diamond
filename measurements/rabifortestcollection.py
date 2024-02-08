import numpy as np

from traits.api import Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group, Label

import logging
import time

from hardware.api import PulseGenerator, TimeTagger, Microwave, MicrowaveD, MicrowaveE, RFSource
import hardware.api as ha
from pulsed import Pulsed
from tools.emod import ManagedJob

from tools.utility import GetSetItemsMixin

class Rabi1( Pulsed ):
    
    """Defines a Rabi measurement."""

    switch      = Enum( 'mw_x','mw_b','mw_y',   desc='switch to use for different microwave source',     label='switch' )
    frequency   = Range(low=1,      high=20e9,  value=1.34193e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power       = Range(low=-100.,  high=10.,   value=-20,      desc='microwave power',     label='power [dBm]',    mode='text', auto_set=False, enter_set=True)

    tau_begin   = Range(low=0., high=1e8,       value=1.5,     desc='tau begin [ns]',  label='tau begin [ns]',   mode='text', auto_set=False, enter_set=True)
    tau_end     = Range(low=1., high=1e8,       value=5000.,     desc='tau end [ns]',    label='tau end [ns]',     mode='text', auto_set=False, enter_set=True)
    tau_delta   = Range(low=1., high=1e6,       value=30.,      desc='delta tau [ns]',  label='delta tau [ns]',   mode='text', auto_set=False, enter_set=True)
    laser       = Range(low=1., high=100000.,   value=3000.,    desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    wait        = Range(low=0., high=100000.,   value=1000.,    desc='wait [ns]',       label='wait [ns]',        mode='text', auto_set=False, enter_set=True)
    meaNum = Int(0)
    tau = Array( value=np.array((0.,1.)) )

    time_bins = Array(value=np.array((0, 1)))
    sequence = Instance(list, factory=list)
    keep_data = Bool(False)
    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        if self.keep_data:
            pass
        else:
            self.meaNum = 0
        Pulsed.apply_parameters(self)
        
    def start_up(self):
        PulseGenerator().Night()
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveD().setOutput(self.power, self.frequency)
        elif self.switch=='mw_f':
            ha.MicrowaveF().setOutput(self.power, self.frequency)
            
    def shut_down(self):
        PulseGenerator().Light()
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

    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()
            if self.run_time >= self.stop_time:
                logging.getLogger().debug('Runtime larger than stop_time. Returning')
                self.state = 'done'
                return

            self.start_up()
            PulseGenerator().Night()
            tagger_0 = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), self.n_laser, 0, 2, 3)
            tagger_1 = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), self.n_laser, 1, 2, 3)

            #tagger_0 = TimeTagger.Pulsed(int(self.n_bins), int(np.round(self.bin_width * 1000)), int(self.n_laser), Int(0), Int(2), Int(3))
            #tagger_1 = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), self.n_laser, Int(1), Int(2), Int(3))
            PulseGenerator().Sequence(self.sequence)
            if PulseGenerator().checkUnderflow():
                logging.getLogger().info('Underflow in pulse generator.')
                PulseGenerator().Night()
                PulseGenerator().Sequence(self.sequence)
                
            
            while self.run_time < self.stop_time:
                start_time = time.time()
                self.thread.stop_request.wait(1.0)
                if self.thread.stop_request.isSet():
                    logging.getLogger().debug('Caught stop signal. Exiting.')
                    break
                if PulseGenerator().checkUnderflow():
                    logging.getLogger().info('Underflow in pulse generator.')
                    PulseGenerator().Night()
                    PulseGenerator().Sequence(self.sequence)
                currentcountdata0 = tagger_0.getData() 
                currentcountdata1 =  tagger_1.getData()
                currentcountdata = currentcountdata1 + currentcountdata0
                
                self.count_data = self.old_count_data + currentcountdata
                self.run_time += time.time() - start_time
                self.meaNum += 1
                # print(meaNum)
            if self.run_time < self.stop_time:
                self.state = 'idle'
            else:
                self.state = 'done'
            del tagger_0
            del tagger_1
            # self.meaNum = 0
            self.shut_down()
            PulseGenerator().Light()

        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'
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
                       title='Rabi for test collection',
                  )