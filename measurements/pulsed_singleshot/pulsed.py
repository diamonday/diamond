# -*- coding: utf-8 -*-

import numpy as np
import time
import logging

import traceback

import hardware.api as ha
from hardware.waveformX import SequenceX, Sin, Idle

from traits.api import (
    Instance,
    Range,
    Float,
    Tuple,
    Bool,
    Array,
    Button,
    on_trait_change,
)

from traitsui.api import (
    View,
    Item,
    Group,
    HGroup,
    VGroup,
    Tabbed,
    VSplit,
    TextEditor,
    EnumEditor,
    Include,
)  # , RangeEditor,
from traitsui.menu import Action, Menu, MenuBar


from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin, GetSetItemsHandler


try:
    from tools.file_utility import save_file
except:
    from traitsui.file_dialog import save_file

import analysis.fitting as fitting

from .common import *
from .ui import LinePlot, Histogram

from .temporary import VERSION, DATE


class PulsedAnalyzerHandler(GetSetItemsHandler):

    def save_all(self, info):
        filename = save_file(title="Save All")

        if not filename:
            return

        info.object.save(filename + ".pys")

        try:
            info.object.save(filename + ".pyz")
        except:
            pass

        info.object.save_line_plot(filename + "_lineplot.png")
        info.object.save_histogram(filename + "_histogram.png")

    def save_line_plot(self, info):
        filename = save_file(title="Save Line Plot")
        if not filename:
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_line_plot(filename)

    def save_histogram(self, info):
        filename = save_file(title="Save Histogram")
        if not filename:
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_histogram(filename)


class Pulsed_SingleShot(ManagedJob, GetSetItemsMixin):
    # Measurement Name
    measurement_name = "Pulsed"

    # Single or Dual Measurement
    dual = False

    # Enable Check Statement
    idle_when = 'state in ("idle", "done", "error")'

    get_set_items = ["__doc__", "measurement_name", "dual"]

    # Temporary
    VERSION = VERSION
    DATE = DATE

    get_set_items.extend(["VERSION", "DATE"])

    # Hardware Definitions
    mw_source = ha.Microwave()
    time_tagger = ha.TimeTagger
    pg = ha.PulseGenerator()
    awg = ha.AWG70k()

    pg_ch_triga = "awgA"
    pg_ch_trigb = "awg"
    pg_ch_mwblock = "awg_dis"
    pg_ch_laser = "green"
    pg_ch_tt_laser = "laser"
    pg_ch_tt_seq = "sequence"

    tt_ch_apd = 0
    tt_ch_laser = 2
    tt_ch_seq = 3

    # Channels used in the measurement, both channels are used in general. The variable is used in sequence loading.
    # To be overriden by measurements
    awg_channel = 11

    # AWG Detailed Parameters
    awg_pulse_length = Range(
        low=0.0,
        high=10000.0,
        value=21.0,
        desc="Trigger Pulse Length for triggering AWG",
        label="Trigger Pulse Length [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    awg_trigger_delay = Range(
        low=0.0,
        high=1e6,
        value=1500.0,
        desc="Delay of AWG from Trigger to output",
        label="Trigger Delay [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    awg_ch1_vpp = Range(
        low=0.0,
        high=0.5,
        value=0.5,
        desc="AWG Ch1 Vpp",
        label="AWG Ch1 Vpp [V]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    awg_ch2_vpp = Range(
        low=0.0,
        high=0.5,
        value=0.5,
        desc="AWG Ch2 Vpp",
        label="AWG Ch2 Vpp [V]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    awg_sampling_rate = Range(
        low=1.5e3,
        high=25.0e9,
        value=25.0e9,
        desc="AWG Sampling Rate",
        label="AWG Sampling Rate [S/s]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    awg_extra_view = HGroup(
        Item("awg_ch1_vpp", width=-50, enabled_when=idle_when),
        Item("awg_ch2_vpp", width=-50, enabled_when=idle_when),
        Item("awg_sampling_rate", width=-80, enabled_when=idle_when),
        Item("awg_pulse_length", width=-50, enabled_when=idle_when),
        Item("awg_trigger_delay", width=-50, enabled_when=idle_when),
        label="AWG Settings",
        show_border=True,
    )

    get_set_items.extend(
        [
            "awg_pulse_length",
            "awg_trigger_delay",
            "awg_ch1_vpp",
            "awg_ch2_vpp",
            "awg_sampling_rate",
        ]
    )

    # Job Control
    resubmit_button = Button(
        label="resubmit",
        desc="Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.",
    )
    run_time = Float(value=0.0, label="run time [s]", format_str="%.f")
    stop_time = Range(
        low=1.0,
        value=np.inf,
        desc="Time after which the experiment stops by itself [s]",
        label="Stop time [s]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    job_view = HGroup(
        Item("submit_button", show_label=False, enabled_when=idle_when),
        Item("remove_button", show_label=False),
        Item("resubmit_button", show_label=False, enabled_when=idle_when),
        Item("priority"),
        Item("state", style="readonly"),
        Item("run_time", style="readonly", format_str="%i"),
        Item("stop_time"),
    )

    get_set_items.extend(["run_time", "stop_time"])

    # AWG Sequence Loading
    reload_awg = Bool(True, label="reload", desc="Compile waveforms upon start up.")
    wfm_button = Button(
        label="Load", desc="Compile waveforms and upload them to the AWG."
    )
    upload_progress = Float(
        label="Upload progress", desc="Progress uploading waveforms", mode="text"
    )

    awg_view = HGroup(
        Item("reload_awg", enabled_when=idle_when),
        Item("wfm_button", show_label=False, enabled_when=idle_when),
        Item("upload_progress", style="readonly", format_str="%i"),
        label="AWG",
        show_border=True,
    )

    # Swap and Readout
    swap_mw_freq = Range(
        low=1e9,
        high=10e9,
        value=2.87e9,
        desc="MW Frequency",
        label="MW Freq [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    swap_mw_amp = Range(
        low=0.0,
        high=1.0,
        value=0.1,
        desc="MW Amplitude",
        label="MW Amplitude",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    swap_mw_t_pi = Range(
        low=0.0,
        high=10000.0,
        value=100.0,
        desc="MW Pi",
        label="MW pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    swap_rf_freq = Range(
        low=1e3,
        high=100e6,
        value=3e6,
        desc="RF Frequency",
        label="RF Freq [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    swap_rf_amp = Range(
        low=0.0,
        high=1.0,
        value=0.1,
        desc="RF Power",
        label="RF Amplitude",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    swap_rf_t_pi = Range(
        low=0.0,
        high=1e9,
        value=30000,
        desc="RF Pi",
        label="RF pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    read_num = Range(
        low=1,
        high=100000,
        value=3000,
        desc="Number of Readout Repetitions",
        label="Read Number",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    read_count_low = Range(
        low=0,
        value=200,
        high=1000,
        desc="Min Count",
        label="Min Count",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    read_count_high = Range(
        low=200,
        value=400,
        high=2000,
        desc="Max Count",
        label="Max Count",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    read_count_delta = Range(
        low=1,
        value=5,
        hight=100,
        desc="Delta Count",
        label="Delta Count",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    enhanced_readout = Bool(
        False,
        label="Enhanced Readout",
        desc="Additionally record data for enhanced_readout",
    )

    read_laser_delay = Range(
        low=-1000.0,
        high=1000.0,
        value=0.0,
        desc="Laser Delay [ns]",
        label="Laser Delay [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    read_start_pos = Range(
        low=0.0,
        desc="Read Start [ns]",
        label="Read Start [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    read_end_pos = Range(
        low=300.0,
        desc="Read End [ns]",
        label="Read End [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    laser_length = Range(
        low=0,
        high=10000,
        value=300,
        desc="Laser Length",
        label="Laser Length [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    wait_length = Range(
        low=0,
        high=10000,
        value=1000,
        desc="Wait Length",
        label="Wait Length [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    record_length = Range(
        low=0,
        high=10000,
        value=350,
        desc="Record Length",
        label="Record Length [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    bin_width = Range(
        low=0.1,
        high=1000.0,
        value=10.0,
        desc="Bin Width",
        label="Bin Width [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    # Nuclear Spin State Reset
    # 	mix_t = Range(low=0.0, high=1e8, desc="Mixing Time After Readout", label="Mix Time [ns]")

    readout_view = Group(
        HGroup(
            Item("swap_mw_freq", enabled_when=idle_when),
            Item("swap_mw_amp", enabled_when=idle_when),
            Item("swap_mw_t_pi", enabled_when=idle_when),
            Item("swap_rf_freq", enabled_when=idle_when),
            Item("swap_rf_amp", enabled_when=idle_when),
            Item("swap_rf_t_pi", enabled_when=idle_when),
        ),
        HGroup(
            Item("read_num", enabled_when=idle_when),
            Item("read_count_low", enabled_when=idle_when),
            Item("read_count_high", enabled_when=idle_when),
            Item("read_count_delta", enabled_when=idle_when),
            Item("enhanced_readout", enabled_when=idle_when),
        ),
        HGroup(
            Item("read_laser_delay", enabled_when=idle_when),
            Item("read_start_pos", enabled_when=idle_when),
            Item("read_end_pos", enabled_when=idle_when),
            Item("laser_length", enabled_when=idle_when),
            Item("wait_length", enabled_when=idle_when),
            Item("record_length", enabled_when=idle_when),
            Item("bin_width", enabled_when=idle_when),
        ),
        label="Swap and Readout",
        show_border=True,
    )

    get_set_items.extend(
        [
            "swap_mw_freq",
            "swap_mw_amp",
            "swap_mw_t_pi",
            "swap_rf_freq",
            "swap_rf_amp",
            "swap_rf_t_pi",
            "read_num",
            "read_count_low",
            "read_count_high",
            "read_count_delta",
            "enhanced_readout",
            "read_laser_delay",
            "read_start_pos",
            "read_end_pos",
            "laser_length",
            "wait_length",
            "record_length",
            "bin_width",
        ]
    )

    init_laser_length = Range(
        low=0,
        high=10000,
        value=1000,
        desc="Initialization Laser Length",
        label="Initialization Laser Length [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    diagnostic_view = Group(
        HGroup(
            Item("init_laser_length", enabled_when=idle_when),
        ),
        label="Diagnostic Settings",
        show_border=True,
    )

    get_set_items.extend(["init_laser_length"])

    datapoints_selector = Instance(
        DataPointsSelector, factory=LinearPoints, args=((0.0, 20000.0), "ns")
    )

    get_set_items.extend(["datapoints_selector"])

    # Plots Components
    lineplot_dark_threshold = Range(
        low=0,
        high=1000,
        value=300,
        label="Dark Threshold",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    lineplot_bright_threshold = Range(
        low=0,
        high=1000,
        value=280,
        label="Bright Threshold",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    lineplot_enhancedata = Bool(desc="Enhanced Readout", label="Enhanced Readout")
    lineplot = Instance(LinePlot, factory=LinePlot)

    lineplot_view = Group(
        HGroup(
            Item("lineplot_dark_threshold"),
            Item("lineplot_bright_threshold"),
            Item("lineplot_enhancedata"),
        ),
        Item("lineplot", style="custom", show_label=False, resizable=True),
    )

    get_set_items.extend(
        ["lineplot_dark_threshold", "lineplot_bright_threshold", "lineplot_enhancedata"]
    )

    histogram_index = Range(
        low=1,
        value=1,
        desc="Index for Datapoint",
        label="Index",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    histogram_datapoint = Float(desc="Datapoint Point", label="Datapoint")
    histogram = Instance(Histogram, factory=Histogram, kw={"dual": dual})

    histogram_view = Group(
        HGroup(Item("histogram_index"), Item("histogram_datapoint", style="readonly")),
        Item("histogram", style="custom", show_label=False, resizable=True),
        show_labels=False,
        label="Histogram",
    )

    get_set_items.extend(["histogram_index", "histogram_datapoint"])

    visualization_view = Tabbed(
        Group(
            Include("lineplot_view"),
            label="Line Plot",
            show_labels=False,
        ),
        Include("histogram_view"),
    )

    # Dummy Values
    runs = 0
    time_bins = Array()
    datapoints = Array()
    data_bins = Array()
    count_data = Array()
    spin_state = Array()
    enhanced_data = Array()
    sequence = []
    sequence_time = 0
    keep_data = False

    get_set_items.extend(
        [
            "runs",
            "time_bins",
            "datapoints",
            "data_bins",
            "count_data",
            "spin_state",
            "enhanced_data",
            "sequence",
            "sequence_time",
            "keep_data",
        ]
    )

    # To be overriden by actual measurement
    measurement_view = None

    measurement_items = []

    get_set_items.extend(measurement_items)

    menu_bar = MenuBar(
        Menu(
            Action(action="save_all", name="Save All"),
            Action(action="save", name="Save (.pyd or .pys)"),
            Action(action="load", name="Load (.pyd or .pys)"),
            Action(action="save_line_plot", name="Save Line Plot (.png)"),
            Action(action="save_histogram", name="Save Histogram (.png)"),
            Action(action="_on_close", name="Quit"),
            name="File",
        )
    )

    traits_view = View(
        Group(
            Include("job_view"),
            VSplit(
                Tabbed(
                    VGroup(
                        Include("awg_view"),
                        Include("readout_view"),
                        Group(
                            Item(
                                "datapoints_selector",
                                style="custom",
                                show_label=False,
                                enabled_when=idle_when,
                            ),
                            Include("measurement_view"),
                            label="Measurement",
                            show_border=True,
                        ),
                        label="Main",
                    ),
                    Group(
                        Include("awg_extra_view"),
                        Include("diagnostic_view"),
                        label="Settings",
                    ),
                ),
                Include("visualization_view"),
            ),
        ),
        menubar=menu_bar,
        handler=PulsedAnalyzerHandler,
        title="Pulsed Analyzer",
        buttons=[],
        resizable=True,
    )

    def __init__(self):
        super(Pulsed_SingleShot, self).__init__()

        self.awg.sync_upload_trait(self, "upload_progress")

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

    # Plot Saving Functions
    def save_matrix_plot(self, filename):
        pass

    def save_line_plot(self, filename):
        self.save_figure(self.lineplot.plot, filename)

    def save_histogram(self, filename):
        self.save_figure(self.histogram.plot, filename)

    def save_pulse_plot(self, filename):
        pass

    # Data Processing Functions

    # Line Plot Related Functions
    @on_trait_change("datapoints")
    def update_datapoints_changed(self):
        self.lineplot.update_data("x", self.datapoints)

    @on_trait_change(
        "count_data, lineplot_dark_threshold, lineplot_bright_threshold, lineplot_enhancedata"
    )
    def update_spin_state(self):
        if len(self.datapoints) == 0 or len(self.count_data) == 0:
            return

        if self.enhanced_readout and self.lineplot_enhancedata:
            self.spin_state = self.enhanced_data

            self.trait_property_changed("spin_state", self.spin_state)
            return

        if self.lineplot_dark_threshold <= self.read_count_low:
            _low_count = np.zeros(self.datapoints.size, dtype=np.float)
        else:
            _thres = (
                int(
                    (self.lineplot_dark_threshold - self.read_count_low)
                    // self.read_count_delta
                )
                + 1
            )
            _low_count = self.count_data[:, 1:_thres].sum(1, dtype=np.float)

        if self.lineplot_bright_threshold >= self.read_count_high:
            _high_count = np.zeros(self.datapoints.size, dtype=np.float)
        else:
            _thres = (
                int(
                    (self.lineplot_bright_threshold - self.read_count_low)
                    // self.read_count_delta
                )
                + 1
            )
            _high_count = self.count_data[:, _thres:-1].sum(1, dtype=np.float)

        _total = _low_count + _high_count
        self.spin_state = _high_count / _total

    @on_trait_change("spin_state")
    def update_spin_state_changed(self):
        _len = len(self.datapoints)

        self.lineplot.update_data("y1", self.spin_state[:_len])
        if self.dual:
            self.lineplot.update_data("y2", self.spin_state[_len : _len * 2])

    # Histogram Related Functions
    @on_trait_change("histogram_index, data_bins, datapoints")
    def update_histogram(self):
        if len(self.datapoints) == 0 or len(self.data_bins) == 0:
            return

        _len = len(self.datapoints)
        _ind = min(self.histogram_index, _len) - 1

        _loc = self.datapoints[_ind]

        self.histogram_datapoint = _loc

        self.histogram.update_data("x", self.data_bins)

        if self.runs and len(self.count_data):
            self.histogram.update_data("y1", self.count_data[_ind])
            if self.dual:
                self.histogram.update_data("y2", self.count_data[_ind + _len])

    @on_trait_change("count_data")
    def update_count_data_change(self):
        if len(self.datapoints) == 0 or len(self.count_data) == 0:
            return

        _len = len(self.datapoints)
        _ind = min(self.histogram_index, _len) - 1

        self.histogram.update_data("y1", self.count_data[_ind])
        if self.dual:
            self.histogram.update_data("y2", self.count_data[_ind + _len])

    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""

        datapoints = self.datapoints_selector.get_datapoints()

        data_bins = np.arange(
            self.read_count_low,
            self.read_count_high + self.read_count_delta,
            self.read_count_delta,
        )

        n_bins = int(self.record_length / self.bin_width)
        self.n_bins = n_bins

        time_bins = self.bin_width * np.arange(n_bins)
        self.time_bins = time_bins

        depth = int(self.read_num * len(datapoints) * (1 + self.dual))
        self.depth = depth

        if (
            self.keep_data
            and np.all(datapoints == self.datapoints)
            and np.all(self.data_bins == data_bins)
            and np.all(time_bins == self.time_bins)
        ):
            pass
        else:
            self.run_time = 0.0
            self.runs = 0

            self.datapoints = datapoints

            self.data_bins = data_bins
            self.count_data = np.zeros(
                (len(datapoints) * (1 + self.dual), data_bins.size + 2), dtype=np.int32
            )
            self.spin_state = np.zeros(len(datapoints) * (1 + self.dual))

            if self.enhanced_readout:
                self.enhanced_data = np.zeros(len(datapoints) * (1 + self.dual))

        self.sequence = self.generate_sequence()
        self.sequence_time = self.calculate_sequence_time(self.sequence)

        # prepare awg
        if not self.keep_data:
            self.prepare_awg()

        self.keep_data = True

    def calculate_sequence_time(self, sequence):
        _total_time = 0

        for ch, t in sequence:
            _total_time += t

        _total_time /= 1e9

        return _total_time

    def prepare_awg(self):
        if self.reload_awg:
            self.load_wfm()
        self.awg.set_vpp(self.awg_ch1_vpp, 0b01)
        self.awg.set_vpp(self.awg_ch2_vpp, 0b10)
        self.awg.set_sampling(self.awg_sampling_rate)

        self.awg.set_run_mode(self.awg.mode_ch1, channel=0b01)
        self.awg.set_run_mode(self.awg.mode_ch2, channel=0b10)

        self.awg.set_trigger(
            TRIG_source=[self.awg.trigger_ch1, self.awg.trigger_ch2],
            TRIG_edge=[self.awg.trigger_edge_ATR, self.awg.trigger_edge_BTR],
            TRIG_level=[
                self.awg.trigger_level_ATR,
                self.awg.trigger_level_ATR,
            ],  # Copied from pulsed_awg1. Bug?
        )

    def _wfm_button_fired(self):
        self.reload_awg = True
        # Trigger a update of parameters
        self.apply_parameters()
        self.prepare_awg()

    def load_wfm(self):
        self.awg_waves = []
        self.awg_main_wave = ""

        self.generate_waveform()

        self.awg.upload(self.awg_waves)
        self.awg.makeseqx(self.awg_main_wave, self.awg_waves)

        if self.awg_channel == 0o1:
            self.awg.managed_load(self.awg_main_wave, 1)
        elif self.awg_channel == 10:
            self.awg.managed_load(self.awg_main_wave, 2)
        elif self.awg_channel == 11:
            self.awg.managed_load(self.awg_main_wave, 11)
        else:
            pass

        self.reload_awg = False

    def generate_sequence_swap(self, index, point):
        _sequence = [
            ((self.pg_ch_trigb,), self.awg_pulse_length),
            (
                tuple(),
                self.awg_trigger_delay + self.swap_mw_t_pi * 2 + self.swap_rf_t_pi,
            ),
            ((self.pg_ch_laser,), self.init_laser_length),
            (tuple(), self.wait_length),
        ]

        return _sequence

    def generate_waveform_swap(self, index, point):

        sampling = self.awg_sampling_rate

        mw_freq = self.swap_mw_freq / sampling
        rf_freq = self.swap_rf_freq / sampling

        mw_t_pi = self.swap_mw_t_pi * (sampling / 1e9)
        rf_t_pi = self.swap_rf_t_pi * (sampling / 1e9)

        rf_T = sampling / self.swap_rf_freq

        rf_cycle = int(rf_t_pi / rf_T)

        if rf_T < 2400:
            logging.getLogger().warning(
                "Initialization RF Cycle shorter than 2400 points"
            )

        mw_R_x = Sin(mw_t_pi, freq=mw_freq, amp=self.swap_mw_amp, phase=0.0)
        idle = Idle(mw_t_pi)

        mw_wavename1 = "SWAP1_%i_1" % index
        rf_wavename1 = "SWAP2_%i_1" % index

        mw_sequence1 = [mw_R_x]
        rf_sequence1 = [idle]

        if mw_t_pi < 2400:
            _idle = 2400 - mw_t_pi
            _fill_sequence = Idle(_idle)

            mw_sequence1.append(_fill_sequence)
            rf_sequence1.append(_fill_sequence)

        self.awg_sequence.addWaveform(mw_wavename1, mw_sequence1)
        self.awg_sequence.addWaveform(rf_wavename1, rf_sequence1)

        self.awg_sequence.seqBox[self.measurement_name].writeStep(
            (mw_wavename1, "Waveform"),
            (rf_wavename1, "Waveform"),
            WaitInput="TrigB",
            Repeat="Once",
            EventJumpInput="None",
            GoTo="Next",
        )

        self.awg_waves.append(self.awg_sequence.wavBox[mw_wavename1])
        self.awg_waves.append(self.awg_sequence.wavBox[rf_wavename1])

        self.awg_sequence.meas_curstep += 1

        rf_R_x = Sin(rf_T, freq=rf_freq, amp=self.swap_rf_amp, phase=0.0)
        idle = Idle(rf_T)

        mw_wavename2 = "SWAP1_%i_2" % index
        rf_wavename2 = "SWAP2_%i_2" % index

        mw_sequence2 = [idle]
        rf_sequence2 = [rf_R_x]

        self.awg_sequence.addWaveform(mw_wavename2, mw_sequence2)
        self.awg_sequence.addWaveform(rf_wavename2, rf_sequence2)

        self.awg_sequence.seqBox[self.measurement_name].writeStep(
            (mw_wavename2, "Waveform"),
            (rf_wavename2, "Waveform"),
            WaitInput="None",
            Repeat=rf_cycle,
            EventJumpInput="None",
            GoTo="Next",
        )

        self.awg_waves.append(self.awg_sequence.wavBox[mw_wavename2])
        self.awg_waves.append(self.awg_sequence.wavBox[rf_wavename2])

        self.awg_sequence.meas_curstep += 1

        mw_wavename3 = "SWAP1_%i_3" % index
        rf_wavename3 = "SWAP2_%i_3" % index

        self.awg_sequence.addWaveform(mw_wavename3, mw_sequence1)
        self.awg_sequence.addWaveform(rf_wavename3, rf_sequence1)

        self.awg_sequence.seqBox[self.measurement_name].writeStep(
            (mw_wavename3, "Waveform"),
            (rf_wavename3, "Waveform"),
            WaitInput="None",
            Repeat="Once",
            EventJumpInput="None",
            GoTo="Next",
        )

        self.awg_waves.append(self.awg_sequence.wavBox[mw_wavename3])
        self.awg_waves.append(self.awg_sequence.wavBox[rf_wavename3])

        self.awg_sequence.meas_curstep += 1

    def generate_sequence_readout(self, index, point):

        # Since the signal to TT is likely only a trigger, the length does not matter as much
        if self.read_laser_delay <= 0:
            _advance_ch = self.pg_ch_laser
            self.pg_ch_tt_laser

            _delay = -self.read_laser_delay
            _laser_length = self.laser_length - _delay
        else:
            _advance_ch = self.pg_ch_tt_laser
            self.pg_ch_laser

            _delay = self.read_laser_delay
            _laser_length = self.laser_length

        _sequence = [
            ((self.pg_ch_trigb, self.pg_ch_mwblock), self.awg_pulse_length),
            (
                (self.pg_ch_mwblock,),
                self.awg_trigger_delay + 96,
            ),  # 96ns due to the padding at the start
        ]

        _sequence += [
            (tuple(), self.swap_mw_t_pi),
            ((_advance_ch, self.pg_ch_mwblock), _delay),
            (
                (self.pg_ch_laser, self.pg_ch_tt_laser, self.pg_ch_mwblock),
                _laser_length,
            ),
            ((self.pg_ch_mwblock,), self.wait_length),
        ] * self.read_num

        # Leave the read loop
        _sequence += [
            ((self.pg_ch_triga,), self.awg_pulse_length),
            (tuple(), self.awg_trigger_delay + 10000),
        ]

        return _sequence

    def generate_waveform_readout(self, index, point):

        sampling = self.awg_sampling_rate

        idle_pad = Idle(2400)

        mw_barrier_wavename = "READ_BARRIER1_%i" % index
        rf_barrier_wavename = "READ_BARRIER2_%i" % index

        mw_barrier_sequence = [idle_pad]
        rf_barrier_sequence = [idle_pad]

        self.awg_sequence.addWaveform(mw_barrier_wavename, mw_barrier_sequence)
        self.awg_sequence.addWaveform(rf_barrier_wavename, rf_barrier_sequence)

        self.awg_sequence.seqBox[self.measurement_name].writeStep(
            (mw_barrier_wavename, "Waveform"),
            (rf_barrier_wavename, "Waveform"),
            WaitInput="TrigB",
            Repeat="Once",
            GoTo="Next",
        )

        self.awg_waves.append(self.awg_sequence.wavBox[mw_barrier_wavename])
        self.awg_waves.append(self.awg_sequence.wavBox[rf_barrier_wavename])

        self.awg_sequence.meas_curstep += 1

        mw_freq = self.swap_mw_freq / sampling
        mw_20000T = sampling / self.swap_mw_freq * 20000

        self.swap_mw_t_pi * (sampling / 1e9)

        R_x = Sin(mw_20000T, freq=mw_freq, amp=self.swap_mw_amp, phase=0.0)
        idle = Idle(mw_20000T)

        mw_wavename = "READ1_%i" % index
        rf_wavename = "READ2_%i" % index

        mw_sequence = [R_x]
        rf_sequence = [idle]

        self.awg_sequence.addWaveform(mw_wavename, mw_sequence)
        self.awg_sequence.addWaveform(rf_wavename, rf_sequence)

        self.awg_sequence.seqBox[self.measurement_name].writeStep(
            (mw_wavename, "Waveform"),
            (rf_wavename, "Waveform"),
            WaitInput="None",
            Repeat="Infinite",
            EventJumpInput="TrigA",
            EventJumpTo=self.awg_sequence.meas_curstep + 2,
            GoTo="Next",
        )

        self.awg_waves.append(self.awg_sequence.wavBox[mw_wavename])
        self.awg_waves.append(self.awg_sequence.wavBox[rf_wavename])

        self.awg_sequence.meas_curstep += 1

        idle_pad = Idle(2400)

        mw_pad_wavename = "READ_PAD1_%i" % index
        rf_pad_wavename = "READ_PAD2_%i" % index

        mw_pad_sequence = [idle_pad]
        rf_pad_sequence = [idle_pad]

        self.awg_sequence.addWaveform(mw_pad_wavename, mw_pad_sequence)
        self.awg_sequence.addWaveform(rf_pad_wavename, rf_pad_sequence)

        # Loop Back to the Start after each sequence
        if index == len(self.datapoints) * (1 + self.dual) - 1:
            _goto = 1
        else:
            _goto = "Next"

        self.awg_sequence.seqBox[self.measurement_name].writeStep(
            (mw_pad_wavename, "Waveform"),
            (rf_pad_wavename, "Waveform"),
            WaitInput="None",
            Repeat="Once",
            EventJumpInput="None",
            GoTo=_goto,
        )

        self.awg_waves.append(self.awg_sequence.wavBox[mw_pad_wavename])
        self.awg_waves.append(self.awg_sequence.wavBox[rf_pad_wavename])

        self.awg_sequence.meas_curstep += 1

    def generate_sequence_mix(self, index, point):
        _sequence = []

        return _sequence

    def generate_waveform_mix(self, index, point):
        pass

    # To be overriden by different measurement subclasses
    def generate_sequence_measurement(self, index, point, n):
        return []

    def generate_waveform_measurement(self, index, point, n):
        pass

    def generate_sequence(self):
        sequence = [((self.pg_ch_tt_seq,), 100)]

        # By default, those sequence are the same for every data point
        _swap_seq = self.generate_sequence_swap(0, 0)
        _read_seq = self.generate_sequence_readout(0, 0)
        _mix_seq = self.generate_sequence_mix(0, 0)

        for ind, dp in enumerate(self.datapoints):
            sequence += (
                self.generate_sequence_measurement(ind, dp, 1)
                + _swap_seq
                + _read_seq
                + _mix_seq
            )

        if self.dual:
            _len = len(self.datapoints)
            for ind, dp in enumerate(self.datapoints):
                sequence += (
                    self.generate_sequence_measurement(ind + _len, dp, 2)
                    + _swap_seq
                    + _read_seq
                    + _mix_seq
                )

        # Allows the end of sequence to be detected through TimeTagger
        sequence += [((self.pg_ch_tt_seq,), 100)]

        return sequence

    def generate_waveform(self):
        name = self.measurement_name

        self.awg_sequence = SequenceX(name)
        self.awg_sequence.meas_curstep = 0

        for ind, dp in enumerate(self.datapoints):
            self.generate_waveform_measurement(ind, dp, 1)
            self.generate_waveform_swap(ind, dp)
            self.generate_waveform_readout(ind, dp)
            self.generate_waveform_mix(ind, dp)

        if self.dual:
            _len = len(self.datapoints)
            for ind, dp in enumerate(self.datapoints):
                self.generate_waveform_measurement(ind + _len, dp, 2)
                self.generate_waveform_swap(ind + _len, dp)
                self.generate_waveform_readout(ind + _len, dp)
                self.generate_waveform_mix(ind + _len, dp)

        self.awg_sequence.seqBox[name].compileSML()
        self.awg_waves.append(self.awg_sequence.seqBox[name])
        self.awg_main_wave = self.awg_sequence.name

    def start_up(self):
        self.pg.Night()

        self.awg.set_output(self.awg.channel)
        self.awg.run()
        time.sleep(1)
        self.awg.update_run_state()

        self.measurement_start_up()

    def shut_down(self):
        self.pg.Light()

        self.awg.stop()
        self.awg.set_output(0b00)
        time.sleep(1)
        self.awg.update_run_state()

        self.measurement_shut_down()

    # To be overriden by measurements as needed
    def measurement_start_up(self):
        pass

    def measurement_shut_down(self):
        pass

    def _run(self):

        _exit_state = ["error", "idle", "done", "error"]
        _exit_reason = 0

        try:
            self.state = "run"
            self.apply_parameters()

            _read_start_ind = int(self.read_start_pos // self.bin_width)
            _read_end_ind = int(self.read_end_pos // self.bin_width)

            _wait_time = self.sequence_time * 0.01

            self.start_up()

            _tagger = self.time_tagger.Pulsed(
                self.n_bins,
                int(np.round(self.bin_width * 1000)),
                self.depth,
                self.tt_ch_apd,
                self.tt_ch_laser,
                self.tt_ch_seq,
            )

            # Allows the end of sequence to be detected through TimeTagger
            _tagger.setMaxCounts(2)

            self.pg.Sequence(self.sequence, loop=False)

            time.sleep(0.5)

            while self.run_time < self.stop_time:
                _start_time = time.time()

                _iter = 0

                # Wait for completion of a measurement run
                while _tagger.getCounts() < 2 and _iter < 50001:
                    time.sleep(_wait_time)

                    _iter += 1

                if _iter == 50001:
                    logging.getLogger().warn("Singleshot Measurement Timed out.")
                    _exit_reason = 3

                    break

                _count_data = _tagger.getData()
                _tagger.clear()

                # Temporary
                self.pg.device.start(1)

                _data = _count_data[:, _read_start_ind:_read_end_ind].sum(1)
                _data = _data.reshape(
                    len(self.datapoints) * (1 + self.dual), self.read_num
                ).sum(1)

                # Save data for enhanced readout
                if self.enhanced_readout:
                    self.enhanced_data += _data

                for i in range(0, len(self.datapoints) * (1 + self.dual)):
                    _sum = _data[i]

                    if _sum < self.read_count_low:
                        self.count_data[i][0] += 1
                    elif _sum > self.read_count_high:
                        self.count_data[i][-1] += 1
                    else:
                        _ind = (_sum - self.read_count_low) // self.read_count_delta + 1

                        self.count_data[i][_ind] += 1

                self.runs += 1

                if not self.runs % 20:
                    self.trait_property_changed("count_data", self.count_data)

                self.run_time += time.time() - _start_time

                if self.thread.stop_request.isSet():
                    _exit_reason = 1
                    break

            else:
                _exit_reason = 2

        except Exception:
            traceback.print_exc()
            _exit_reason = 3

        finally:
            self.shut_down()
            del _tagger

        self.state = _exit_state[_exit_reason]


class Rabi(Pulsed_SingleShot):
    measurement_name = "Rabi"

    datapoints_selector = Instance(
        DataPointsSelector, factory=LinearPoints, args=((0.0, 1e6), "ns")
    )

    # Inherit the get_set_items list
    get_set_items = Pulsed_SingleShot.get_set_items + []

    # Measurement specific settings
    mw_freq = Range(
        low=1e9,
        high=10e9,
        value=2.87e9,
        desc="MW Frequency",
        label="MW Freq [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    mw_amp = Range(
        low=0.0,
        high=1.0,
        value=0.1,
        desc="MW Amplitude",
        label="MW Amplitude",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    get_set_items.extend(["mw_freq", "mw_amp"])

    measurement_view = Group(
        HGroup(
            Item("mw_freq", enabled_when=Pulsed_SingleShot.idle_when),
            Item("mw_amp", enabled_when=Pulsed_SingleShot.idle_when),
        )
    )

    # Fitting Settings
    fit_enable = Bool(desc="Enable Fit", label="Enable Fit")
    fit_T = Tuple((0.0, 0.0), desc="Period", label="Period")
    fit_contrast = Tuple((0.0, 0.0), desc="Contrast", label="Contrast")
    fit_view = HGroup(
        Item("fit_enable"),
        Item("fit_T", style="readonly", format_str="%.2f ± %.2f ns"),
        Item("fit_contrast", style="readonly", format_str="%.2f ± %.2f %%"),
    )

    get_set_items.extend(["fit_enable", "fit_T", "fit_contrast"])

    visualization_view = Tabbed(
        Group(
            Include("fit_view"),
            Include("lineplot_view"),
            label="Line Plot",
        ),
        Include("histogram_view"),
    )

    # Functions for fitting
    @on_trait_change("fit_enable, spin_state")
    def fit_data(self):
        if not self.fit_enable or len(self.spin_state) == 0:
            self.lineplot.update_data("fit1", None)
            return
        try:
            amp = 2**0.5 * np.sqrt(
                ((self.spin_state - self.spin_state.mean()) ** 2).sum()
            )

            _f = np.fft.fft(self.spin_state)
            _N = len(_f)
            _D = float(self.datapoints[1] - self.datapoints[0])
            _i = abs(_f[1 : _N / 2 + 1]).argmax() + 1
            _T = (_N * _D) / _i

            fit_result = fitting.fit_rabi_phase(
                self.datapoints, self.spin_state, self.spin_state**0.5, [amp, _T, np.pi]
            )
        except:
            fit_result = (
                np.NaN * np.zeros(4),
                np.NaN * np.zeros((4, 4)),
                np.NaN,
                np.NaN,
            )

        p, v, q, chisqr = fit_result
        a, T, phi, c = p
        a_var, T_var, phi_var, c_var = v.diagonal()

        contrast = 200 * a / (c + a)
        contrast_delta = 200.0 / c**2 * (a_var * c**2 + c_var * a**2) ** 0.5
        T_delta = abs(T_var) ** 0.5

        if self.enhanced_readout and self.lineplot_enhancedata:
            self.fit_T = T, T_delta
            self.fit_contrast = contrast, contrast_delta
        else:
            self.fit_T = T, 0
            self.fit_contrast = contrast, 0

        if fit_result[0][0] != np.NaN:
            self.lineplot.update_data(
                "fit1", fitting.Cosinus_phase(*fit_result[0])(self.datapoints)
            )

    def generate_sequence_measurement(self, index, point, n):
        sequence = [
            ((self.pg_ch_laser,), self.init_laser_length),
            (tuple(), self.wait_length),
            ((self.pg_ch_trigb,), self.awg_pulse_length),
            (tuple(), self.awg_trigger_delay + point),
        ]

        return sequence

    def generate_waveform_measurement(self, index, point, n):

        sampling = self.awg_sampling_rate

        mw_freq = self.mw_freq / sampling

        mw_t_pi = point * (sampling / 1e9)

        mw_R_x = Sin(mw_t_pi, freq=mw_freq, amp=self.mw_amp, phase=0.0)

        idle = Idle(mw_t_pi)

        mw_wavename1 = "RABI1_%i_1" % index
        rf_wavename1 = "RABI2_%i_1" % index

        mw_sequence1 = [mw_R_x]
        rf_sequence1 = [idle]

        if mw_t_pi < 2400:
            _idle = 2400 - mw_t_pi
            _fill_sequence = Idle(_idle)

            mw_sequence1.append(_fill_sequence)
            rf_sequence1.append(_fill_sequence)

        self.awg_sequence.addWaveform(mw_wavename1, mw_sequence1)
        self.awg_sequence.addWaveform(rf_wavename1, rf_sequence1)

        self.awg_sequence.seqBox[self.measurement_name].writeStep(
            (mw_wavename1, "Waveform"),
            (rf_wavename1, "Waveform"),
            WaitInput="TrigB",
            Repeat="Once",
            EventJumpInput="None",
            GoTo="Next",
        )

        self.awg_waves.append(self.awg_sequence.wavBox[mw_wavename1])
        self.awg_waves.append(self.awg_sequence.wavBox[rf_wavename1])

        self.awg_sequence.meas_curstep += 1
