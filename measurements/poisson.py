import numpy as np
import logging
import time
import os.path
import traceback

from traits.api import HasTraits, String, Range, Int, Str, Float, Bool, Tuple, Array, Instance, Property, Enum, on_trait_change, Button
from traitsui.api import View, Item, Tabbed, HGroup, VGroup, VSplit, EnumEditor, TextEditor
from traitsui.menu import Action, Menu, MenuBar
from traitsui.file_dialog import save_file

from hardware.api import PulseGenerator, TimeTagger, Microwave
from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin, GetSetItemsHandler

from pulsed_singleshot.common import *
from pulsed_singleshot.ui import Histogram


class PoissonHandler(GetSetItemsHandler):

    def save_histogram(self, info):
        filename = save_file(title='Save Histogram')
        if not filename:
            return
        else:
            if filename.find('.png') == -1:
                filename = filename + '.png'
            info.object.save_histogram(filename)

    def saveAll(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            #info.object.save_all_figure(filename)
            info.object.save(filename)
        info.object.save_histogram(filename + '_histogram.png')


class SequenceHandler(HasTraits, GetSetItemsMixin):
    
    state = Enum('idle', 'run', 'wait', 'done', 'error')

    # Specify which items to be saved
    get_set_items = []

    # Microwave parameters
    frequency = Range(
        low=1, high=20e9, value=1.454545e9,
        desc='microwave frequency', label='frequency mw[Hz]',
        mode='text', auto_set=False, enter_set=True,
        editor=TextEditor(evaluate=float, format_str='%e')
    )
    power_mw = Range(
        low= -100., high=16., value=-10,
        desc='power of microwave', label='power mw[dBm]',
        mode='text', auto_set=False, enter_set=True
    )
    t_2pi_x = Range(
        low=1., high=100000., value=200.,
        desc='rabi period 2pi pulse length (x)', label='2pi x [ns]',
        mode='text', auto_set=False, enter_set=True
    )
    get_set_items += ['frequency', 'power_mw', 't_2pi_x']

    # Laser parameters
    laser_length = Range(
        low=0.0,  high=10000.0, value=300.0,
        desc='Laser Length', label='Laser Length [ns]',
        mode='text', auto_set=False, enter_set=True
    )
    wait_length = Range(
        low=0.0,  high=10000.0, value=0.0,
        desc='wait Length', label='wait Length [ns]',
        mode='text', auto_set=False, enter_set=True
    )
    get_set_items += ['laser_length', 'wait_length']

    '''
    The designed sequence is duplicated for N_rep times.
    This is to prevent underflow error for the PG. Since the PG will loop
    the sequence indefinitely anyway, the exact number of N_rep doesn't matter.
    '''
    N_rep = Range(
        low=1, high=1000, value=200,
        desc='number of DD block', label='N Repitition',
        mode='text', auto_set=False, enter_set=True
    )
    
    T_seq_dark = Float(
        0.0, label='Dark Sequence time',
        editor=TextEditor(evaluate=float, format_str='%.1f ns')
    )
    T_seq_bright = Float(
        0.0, label='Bright Sequence time',
        editor=TextEditor(evaluate=float, format_str='%.1f ns')
    )
    get_set_items += ['N_rep', 'T_seq_dark', 'T_seq_bright']


    def __init__(self, **kwargs):
        super(SequenceHandler, self).__init__()

    def generate_sequence(self):
        # Green laser length
        laser = self.laser_length

        # Wait duration
        wait = self.wait_length  

        # MW time
        t_2pi_x = self.t_2pi_x
        t_pi_x = t_2pi_x/2.0 

        seq_dark = [
            (['A','B','mw_x'], t_pi_x),
            (['B','aom'], laser),
            (['B'], wait),
        ]

        seq_bright = [
            (['B'], t_pi_x),
            (['B','aom'], laser),
            (['B'], wait),
        ]

        self.T_seq_dark = t_pi_x + laser + wait
        self.T_seq_bright = t_pi_x + laser + wait

        return seq_dark*self.N_rep, seq_bright*self.N_rep

    def start_up(self):
        PulseGenerator().Night()
        Microwave().setOutput(self.power_mw, self.frequency)

    def shut_down(self):
        PulseGenerator().Light()
        Microwave().setOutput(None, self.frequency)
    
    traits_view = View(
        VGroup(
            VGroup(
                Item('laser_length', width= -80, enabled_when='state != "run"'),
                Item('wait_length', width= -80, enabled_when='state != "run"'),
                label='Laser',
                show_border = True,
            ),
            VGroup(
                Item('frequency', width= -80, enabled_when='state != "run"'),
                Item('power_mw', width= -80, enabled_when='state != "run"'),
                Item('t_2pi_x', width= -80, enabled_when='state != "run"'),
                Item('N_rep', width= -80, enabled_when='state != "run"'),
                label='Microwave',
                show_border = True,
            ),
            VGroup(
                Item('T_seq_dark', width= -80, style='readonly'),
                Item('T_seq_bright', width= -80, style='readonly'),
                show_border = True,
            ),
            label='Sequence Parameters',
            show_border=True,
        )
    )


class Poisson(ManagedJob, GetSetItemsMixin):
    
    time_tagger = TimeTagger
    pg = PulseGenerator()

    # Specify which items to be saved
    get_set_items = []

    # Reduced time for calling pg.Sequence()
    # Not essential for short sequences
    use_stored_sequence = Bool(False, desc="Save time for generating binary sequence", label="Use Stored Sequence")
    sequence_path = String("", desc="Sequence folder", label="Sequence Folder")
    updateSeq = Button(desc="Update current sequence", label="Update")
    BinSeq = ['','']
    get_set_items += [
        'use_stored_sequence','sequence_path',
    ]

    # A class to store all of the necessary parameters of the sequence,
    # provide GUI, and the methods to generate the sequence itself
    Sequence = Instance(SequenceHandler, factory=SequenceHandler)
    get_set_items += ['Sequence']

    # APD Readout
    read_time = Range(low=0.0001, high=1, value=0.005, desc="Read Time [s]", label="Read Time [s]", mode='text', auto_set=False, enter_set=True)
    read_runs_round = Range(low=1, high=100000, value=200, desc="Read Runs Per Round", label="Runs Per Round", mode='text', auto_set=False, enter_set=True)
    read_delay = Range(low=0.0, high=100.0, value=1.0, desc="Read Delay [s]", label="Read Delay [s]", mode='text', auto_set=False, enter_set=True)
    get_set_items += [
        'read_time','read_runs_round','read_delay',
    ]

    read_count_low = Range(low=0, value=1, high=20000, desc="Min Count", label="Min Count", mode='text', auto_set=False, enter_set=True)
    read_count_high = Range(low=1, value=10, high=20000, desc="Max Count", label="Max Count", mode='text', auto_set=False, enter_set=True)
    read_count_delta = Range(low=1, value=1, hight=100, desc="Delta Count", label="Delta Count", mode='text', auto_set=False, enter_set=True)
    save_time_trace = Bool(False, desc="Save Time Trace", label="Save Time Trace")
    time_trace_path = String("", desc="Time Trace Folder", label="Time Trace Folder")
    get_set_items += [
        'read_count_low','read_count_high','read_count_delta',
        'save_time_trace', 'time_trace_path',
    ]

    # Job Control
    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')
    run_time  = Float(value=0.0, label='run time [s]',format_str='%.f')
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)

    # Visualization
    histogram = Instance(Histogram, factory=Histogram, kw={"dual": True})

    # Dummy Values
    runs = 0
    data_bins = Array()
    count_occurence = Array()
    sequence = []
    keep_data = False

    get_set_items += [
        'runs', 'data_bins', 'count_occurence',
        'sequence', 'keep_data',
    ]

    def __init__(self):
        super(Poisson, self).__init__()
        self.sync_state(self.Sequence)

    # Job Control Functions
    def submit(self):
        """Submit the job to the JobManager."""
        self.keep_data = False
        ManagedJob.submit(self)

    def resubmit(self):
        """Submit the job to the JobManager."""
        self.keep_data = True
        ManagedJob.submit(self)

    def _resubmit_button_fired(self):
        """React to start button. Submit the Job."""
        self.resubmit()

    #==========================================================|
    #               check parameters, set devices              |
    #==========================================================|
    def start_up(self):
        self.Sequence.start_up()

    def shut_down(self):
        self.Sequence.shut_down()

    def generate_sequence(self):
        return self.Sequence.generate_sequence()

    def apply_parameters(self):
        # Record the occurence of each number of counts
        data_bins = np.arange(self.read_count_low, self.read_count_high + self.read_count_delta, self.read_count_delta)
        count_occurence = np.zeros((2,data_bins.size + 2))

        if self.keep_data and np.all(data_bins == self.data_bins): # if the count binning is the same as previous, keep existing data
            pass
        else:
            self.run_time = 0.0
            self.runs = 0
            self.data_bins = data_bins
            self.count_occurence = count_occurence

        if self.use_stored_sequence:
            with open(self.sequence_path + '\BinSeq0.bin', 'rb') as bfile:
                self.BinSeq[0] = bfile.read()
            with open(self.sequence_path + '\BinSeq1.bin', 'rb') as bfile:
                self.BinSeq[1] = bfile.read()
        else:  
            self.sequence = self.generate_sequence()

        # Decide whether to keep previous data
        self.keep_data = True
    
    def load_sequence(self, i):
        if self.use_stored_sequence:
            self.pg.halt()
            self.pg.loadPages(self.BinSeq[i])
            self.pg.run(triggered=False)
        else:
            self.pg.Sequence(self.sequence[i], loop=True)
    
    def collect_counts(self, _counter, i):
        self.load_sequence(i)

        # Wait for the APD to record data
        time.sleep(self.read_delay)
        time.sleep(self.read_time * self.read_runs_round * 1.1)
        
        # Acquire date from APD
        _data = _counter.getData()

        # Save time trace
        if self.save_time_trace:
            _path = os.path.normpath(self.time_trace_path) + "/" + "Dark_%i.npy" % self.runs
            np.save(_path, _data)

        # Assign the number of counts into each data bin
        for n in range(0, self.read_runs_round):
            _sum = _data[n]
            if _sum < self.read_count_low:
                self.count_occurence[i][0] += 1
            elif _sum > self.read_count_high:
                self.count_occurence[i][-1] += 1
            else:
                _ind = (_sum - self.read_count_low) // self.read_count_delta
                self.count_occurence[i][_ind + 1] += 1

    def _run(self):
        _exit_state = ['error', 'idle', 'done', 'error']
        _exit_reason = 0
        ch_apd = 0

        try:
            self.state = 'run'
            self.apply_parameters()
            self.start_up()

            # Set the APD to be the "Counter" mode
            _counter = self.time_tagger.Counter(ch_apd, int(self.read_time * 1e12), self.read_runs_round)

            while self.run_time < self.stop_time:
                
                _start_time = time.time()
                
                ######################### Dark ##########################
                self.collect_counts(_counter, 0)
                #########################################################

                ######################## Bright #########################
                self.collect_counts(_counter, 1)
                #########################################################
                self.runs += 1
                self.trait_property_changed('count_occurence', self.count_occurence)
                self.run_time += time.time() - _start_time

                if self.thread.stop_request.isSet():
                    _exit_reason = 1
                    break

            else:
                _exit_reason = 2

        except Exception as e:
            traceback.print_exc()
            _exit_reason = 3

        finally:
            self.shut_down()
            del _counter

        self.state = _exit_state[_exit_reason]


    #==========================================================|
    #          treat raw data and store data in objects        |
    #==========================================================|
    def sync_state(self, client):
        self.sync_trait('state', client, mutual=False)
        
    @on_trait_change("data_bins")
    def update_histogram(self):
        if len(self.data_bins) == 0:
            return

        self.histogram.update_data("x", self.data_bins)
        self.histogram.plot1.bar_width = self.read_count_delta
        self.histogram.plot2.bar_width = self.read_count_delta

        if self.runs and len(self.count_occurence):
            self.histogram.update_data("y1", self.count_occurence[0])
            self.histogram.update_data("y2", self.count_occurence[1])
        
    @on_trait_change("count_occurence")
    def update_count_occurence_change(self):
        if len(self.count_occurence) == 0:
            return
        self.histogram.update_data("y1", self.count_occurence[0])
        self.histogram.update_data("y2", self.count_occurence[1])

    def _updateSeq_fired(self):
        self.sequence = self.generate_sequence()
        BinSeq0 = self.pg.convertSequenceToBinary(self.sequence[0],loop=True)
        BinSeq1 = self.pg.convertSequenceToBinary(self.sequence[1],loop=True)
        with open(self.sequence_path + '\BinSeq0.bin', 'wb') as bfile:
            bfile.write(BinSeq0)
        with open(self.sequence_path + '\BinSeq1.bin', 'wb') as bfile:
            bfile.write(BinSeq1)
        print('Update Succeed!')

    # Plot Saving Functions
    def save_histogram(self, filename):
        self.save_figure(self.histogram.plot, filename)

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
                    VGroup(
                        Item('read_time', width= -80, enabled_when='state != "run"'),
                        Item('read_runs_round', width= -80, enabled_when='state != "run"'),
                        Item('read_count_low', width= -80, enabled_when='state != "run"'),
                        Item('read_count_high', width= -80, enabled_when='state != "run"'),
                        Item('read_count_delta', width= -80, enabled_when='state != "run"'),
                        label='Photon Counter',
                        show_border = True,
                    ),
                    Item('Sequence', style='custom', show_label=False)
                ),
                VGroup(
                    VGroup(
                        HGroup(
                            Item("use_stored_sequence", enabled_when='state != "run"'),                        
                            Item("sequence_path", width= .1, enabled_when='state != "run"'),
                            Item("updateSeq", enabled_when='state != "run"'),
                        ),
                        HGroup(
                            Item("save_time_trace", enabled_when='state != "run"'),
                            Item("time_trace_path", width= .1, enabled_when='state != "run"'),
                        ),
                        show_border=True,
                    ),
                    Group(
                        Item("histogram", style="custom", show_label=False),
                        label="Histogram",
                        show_border=True,
                    ),
                )
            ),
        ),
        menubar=MenuBar(
            Menu(
                Action(action='load', name='Load'),
                Action(action='save', name='Save (.pyd or .pys)'),
                Action(action='save_histogram', name='Save Histogram (.png)'),
                Action(action='saveAll', name='Save All (.png+.pys)'),
                Action(action='_on_close', name='Quit'),
                name='File',
            ),
        ),
        title='Poisson Histogram',
        width=1250,  
        resizable=True,
        handler=PoissonHandler,
    )

if __name__ == '__main__':
    pass