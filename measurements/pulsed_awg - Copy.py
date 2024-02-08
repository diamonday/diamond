from tokenize import String
import numpy as np

fft = np.fft.fft
ifft = np.fft.ifft

from hardware.waveform import *

from traits.api import (
    Instance,
    String,
    Range,
    Float,
    Int,
    Bool,
    Array,
    Button,
    on_trait_change,
)
from traitsui.api import (
    View,
    Item,
    HGroup,
    VGroup,
    Tabbed,
    VSplit,
    TextEditor,
)  # , EnumEditor, RangeEditor,
from traitsui.menu import Action, Menu, MenuBar
from traitsui.file_dialog import save_file

from enable.api import ComponentEditor
from chaco.api import (
    ArrayDataSource,
    LinePlot,
    LinearMapper,
    ArrayPlotData,
    Plot,
)
from chaco.tools.api import PanTool
from chaco.tools.simple_zoom import SimpleZoom

import logging
import time

import analysis.fitting as fitting

from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin, GetSetItemsHandler
from tools.color import scheme

# # matplotlib stuff
# from tools.utility import MPLFigureEditor
# from matplotlib.figure import Figure as MPLFigure
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import IndexLocator

# Handler


def find_laser_pulses(sequence):
    n = 0
    prev = []
    for channels, t in sequence:
        if "laser" in channels and not "laser" in prev:
            n += 1
        prev = channels
        if ("sequence" in channels) and (n > 0):
            break
    return n


class PulsedToolHandler(GetSetItemsHandler):
    def save_matrix_plot(self, info):
        filename = save_file(title="Save Matrix Plot")
        if filename is "":
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_matrix_plot(filename)

    def save_line_plot(self, info):
        filename = save_file(title="Save Line Plot")
        if filename is "":
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_line_plot(filename)

    def save_fft_plot(self, info):
        filename = save_file(title="Save FFT Plot")
        if filename is "":
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_fft_plot(filename)

    def save_pulse_plot(self, info):
        filename = save_file(title="Save Pulse Plot")
        if filename is "":
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_pulse_plot(filename)


class Pulsed(ManagedJob, GetSetItemsMixin):
    """Base class for measurements running with laser and microwave sequences."""

    # Directory storing .WFM/.PAT files in AWG
    # Default directory is \waves
    main_dir = String(
        "\waves", desc=".WFM/.PAT folder in AWG", label="Waveform Dir.:", mode="text"
    )

    # Traits ########################
    reload_awg = Bool(True, label="reload", desc="Compile waveforms upon start up.")
    wfm_button = Button(
        label="Load", desc="Compile waveforms and upload them to the AWG."
    )
    upload_progress = Float(
        label="Upload progress", desc="Progress uploading waveforms", mode="text"
    )

    vpp1 = Range(
        low=0.0,
        high=2.0,
        value=2.0,
        label="vpp CH1 [V]",
        desc="Set output voltage peak to peak [V] of CH1.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    vpp2 = Range(
        low=0.0,
        high=2.0,
        value=2.0,
        label="vpp CH2 [V]",
        desc="Set output voltage peak to peak [V] of CH2.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    trig_interval = Range(
        low=11.0,
        high=10000.0,
        value=28.0,
        label="Trig Interval",
        desc="Set the trigger duration for the AWG.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    trig_delay = Range(
        low=11.0,
        high=10000.0,
        value=45.0,
        label="Trig Delay",
        desc="Trigger delay of the AWG.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    sampling = Range(
        low=50.0e3,
        high=1.0e9,
        value=1.0e9,
        label="sampling [Hz]",
        desc="Set the sampling rate of the AWG.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    waves = []

    # Counter
    channel_apd_0 = Int(0)
    channel_apd_1 = Int(1)
    channel_detect = Int(2)
    channel_sequence = Int(3)

    bin_width = Range(
        low=0.1,
        high=1000.0,
        value=1.0,
        desc="bin width [ns]",
        label="bin width [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    depth = Int(2)
    n_bins = Int(2)
    time_bins = Array(value=np.array((0, 1)))

    # Sequence
    laser = Range(
        low=1.0,
        high=5e6,
        value=3000.0,
        desc="laser [ns]",
        label="laser [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    wait = Range(
        low=1.0,
        high=5e6,
        value=1000.0,
        desc="wait [ns]",
        label="wait [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    freq = Range(
        low=1,
        high=20e9,
        value=2.70e9,
        desc="SMIQ frequency",
        label="SMIQ frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    power = Range(
        low=-100.0,
        high=25.0,
        value=-20,
        desc="microwave power",
        label="power [dBm]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    start_time = Range(
        low=1.0,
        high=1e7,
        value=15,
        desc="start time [ns]",
        label="start time [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    end_time = Range(
        low=1.0,
        high=2e7,
        value=500.0,
        desc="end time [ns]",
        label="end time [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    time_step = Range(
        low=1.0,
        high=1e6,
        value=5.0,
        desc="time step [ns]",
        label="time step [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    tau = Array()
    sequence = Instance(list, factory=list)

    # Progress
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

    # sweeps = Range(low=1., high=1e10, value=1e6, desc='number of sweeps', label='sweeps', mode='text', auto_set=False, enter_set=True)
    # elapsed_sweeps = Float(value=0, desc='Elapsed Sweeps [s]', label='Elapsed Sweeps [s]', mode='text')
    # progress = Int(value=0, desc='Progress [%]', label='Progress [%]', mode='text')

    # Data
    keep_data = False
    resubmit_button = Button(
        label="resubmit",
        desc="Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.",
    )

    record_length = Range(
        low=0,
        high=10000,
        value=3000,
        desc="time window for pulse analysis [ns]",
        label="profile length [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    time_window_width = Range(
        low=10,
        high=1000,
        value=300,
        desc="time window for pulse analysis [ns]",
        label="signal window [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    time_window_offset_signal = Range(
        low=0,
        high=1000,
        value=25,
        desc="offset of time window for signal [ns]",
        label="signal at [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    time_window_offset_normalize = Range(
        low=0,
        high=10000,
        value=2000,
        desc="offset of time window for normalization [ns]",
        label="normalize at [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    count_data = Array()
    pulse_profile = Array()
    flank = Float()
    y1 = Array()
    y2 = Array()

    nu = Array()
    yf1 = Array()
    yf2 = Array()
    db = Bool(False, label="log scale")

    dual = Bool(False, label="second data set")

    # Plots
    line_plot_data = Instance(
        ArrayPlotData, transient=True
    )  # transient: do not save this trait
    line_plot = Instance(Plot, transient=True)
    matrix_plot_data = Instance(ArrayPlotData, transient=True)
    matrix_plot = Instance(Plot, transient=True)
    pulse_plot_data = Instance(ArrayPlotData, transient=True)
    pulse_plot = Instance(Plot, transient=True)
    fft_plot_data = Instance(ArrayPlotData, transient=True)
    fft_plot = Instance(Plot, transient=True)

    # get_set_items
    get_set_items = [
        "laser",
        "wait",
        "freq",
        "power",
        "run_time",  # later also sweeps
        "pulse_profile",
        "flank",
        "time_window_width",
        "time_window_offset_signal",
        "time_window_offset_normalize",
        "tau",
        "y1",
        "y2",
        "count_data",
        "nu",
        "yf1",
        "yf2",
        "vpp1",
        "vpp2",
        "sampling",
        "trig_interval",
        "trig_delay",
        "start_time",
        "end_time",
        "time_step",
        "main_dir",
    ]

    # Constructor ########################
    def __init__(self, pulse_generator, time_tagger, microwave, awg, **kwargs):
        super(Pulsed, self).__init__(**kwargs)
        self.pulse_generator = pulse_generator
        self.time_tagger = time_tagger
        self.microwave = microwave
        self.awg = awg
        self.awg.sync_upload_trait(self, "upload_progress")

        self.tau = np.arange(self.start_time, self.end_time, self.time_step)

        self._create_matrix_plot()
        self._create_pulse_plot()
        self._create_line_plot()
        self._create_fft_plot()

    # Measurement ########################
    def submit(self):
        """Start new measurement without keeping data."""
        self.keep_data = False
        ManagedJob.submit(self)

    def resubmit(self):
        """Submit the job to the JobManager."""
        self.keep_data = True
        ManagedJob.submit(self)

    def _resubmit_button_fired(self):
        """React to start button. Submit the Job."""
        self.resubmit()

    def load_wfm(self):
        self.waves = []
        self.main_wave = ""
        self.tau = np.arange(self.start_time, self.end_time, self.time_step)
        self.compile_waveforms()
        self.awg.ftp_cwd = "/main" + self.main_dir
        self.awg.upload(self.waves)
        self.awg.managed_load(self.main_wave, cwd=self.main_dir)
        self.reload_awg = False

    def prepare_awg(self):
        if self.reload_awg:
            self.load_wfm()
        self.awg.set_vpp(self.vpp1, 0b01)
        self.awg.set_vpp(self.vpp2, 0b10)
        self.awg.set_sampling(self.sampling)
        self.awg.set_mode("E")

    def _wfm_button_fired(self):
        self.reload_awg = True
        self.prepare_awg()

    def generate_sequence(self):
        """Override this."""
        return [
            ([], 100),
        ]

    def start_up(self):
        """Override for additional stuff to be executed at start up."""
        self.pulse_generator.Night()
        self.microwave.setOutput(self.power, self.freq)
        self.awg.set_output(0b11)
        self.awg.run()

    def shut_down(self):
        """Override for additional stuff to be executed at shut down."""
        self.pulse_generator.Light()
        self.microwave.setOutput(None, self.freq)
        self.awg.stop()
        self.awg.set_output(0b00)

    def shut_down_finally(self):
        """Override for additional stuff to be executed finally."""
        self.pulse_generator.Light()
        self.microwave.setOutput(None, self.freq)
        self.awg.stop()
        self.awg.set_output(0b00)

    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        self.tau = np.arange(self.start_time, self.end_time, self.time_step)
        n_bins = int(self.record_length / self.bin_width)
        time_bins = self.bin_width * np.arange(n_bins)
        sequence = self.generate_sequence()

        self.depth = find_laser_pulses(self.sequence)

        if (
            self.keep_data
            and sequence == self.sequence
            and np.all(time_bins == self.time_bins)
        ):  # if the sequence and time_bins are the same as previous, keep existing data
            self.old_count_data = self.count_data.copy()
        else:
            self.run_time = 0.0
            self.old_count_data = np.zeros((self.depth, n_bins))

        # prepare awg
        if not self.keep_data:
            self.prepare_awg()  # TODO: if reload: block thread untill upload is complete

        self.sequence = sequence
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.keep_data = True  # when job manager stops and starts the job, data should be kept. Only new submission should clear data.

    def _run(self):
        # new style TODO: include sweeps + abort @ sweep
        try:
            self.state = "run"
            self.pulse_generator.Night()

            self.apply_parameters()

            if self.run_time >= self.stop_time:
                logging.getLogger().debug("Runtime larger than stop_time. Returning")
                self.state = "done"
                return
            self.start_up()

            # set up counters and pulse generator
            if self.dual:
                self.depth *= 2
            if self.channel_apd_0 > -1:
                pulsed_0 = self.time_tagger.Pulsed(
                    self.n_bins,
                    int(np.round(self.bin_width * 1000)),
                    self.depth,
                    self.channel_apd_0,
                    self.channel_detect,
                    self.channel_sequence,
                )
            if self.channel_apd_1 > -1:
                pulsed_1 = self.time_tagger.Pulsed(
                    self.n_bins,
                    int(np.round(self.bin_width * 1000)),
                    self.depth,
                    self.channel_apd_1,
                    self.channel_detect,
                    self.channel_sequence,
                )
            self.pulse_generator.Sequence(self.sequence)
            self.pulse_generator.checkUnderflow()

            # count
            while self.run_time < self.stop_time:
                start_time = time.time()
                self.thread.stop_request.wait(1.0)
                if self.thread.stop_request.isSet():
                    logging.getLogger().debug("Caught stop signal. Exiting.")
                    break
                if self.pulse_generator.checkUnderflow():
                    raise RuntimeError("Underflow in pulse generator.")
                if self.channel_apd_0 > -1 and self.channel_apd_1 > -1:
                    self.count_data = (
                        self.old_count_data + pulsed_0.getData() + pulsed_1.getData()
                    )
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
            self.state = "idle"

    # Processing ########################
    @on_trait_change(
        "count_data, time_window_width, time_window_offset_signal, time_window_offset_normalize"
    )
    def _analyze_count_data(self):
        y = self.count_data
        n = len(y)

        T = self.time_window_width
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
        I = int(np.round(T / dt))
        for i, slot in enumerate(y):
            z0[i] = slot[i0 : i0 + I].mean()
            z1[i] = slot[i1 : i1 + I].mean()

        self.flank = flank
        self.pulse_profile = profile
        if self.dual:
            self.y2 = (z0 / z1)[n / 2 :]
            self.y1 = (z0 / z1)[: n / 2]
        else:
            self.y1 = z0 / z1

    # FFT
    @on_trait_change("y1")
    def update_fft(self):
        fund = 1000.0 / (self.tau[-1] - self.tau[0])  # fundamental mode in MHz
        n = len(self.y1)
        self.nu = np.arange(0.0, fund * (n + 1) / 2.0, fund)  # frequencies of fft

        samples1 = self.y1 - self.y1.mean()  # clear offset
        transform1 = abs(fft(samples1))  # fft and clear phase
        if n % 2 == 0:
            self.yf1 = (
                transform1[: n / 2] + transform1[: n / 2 - 1 : -1]
            )  # sum both directions
        else:
            self.yf1 = transform1[: n / 2] + transform1[: n / 2 : -1]

        if self.dual:  # repeat for second data set
            samples2 = self.y2 - self.y2.mean()
            transform2 = abs(fft(samples2))
            if n % 2 == 0:
                self.yf2 = transform2[: n / 2] + transform2[: n / 2 - 1 : -1]
            else:
                self.yf2 = transform2[: n / 2] + transform2[: n / 2 : -1]

    # Plots ########################
    # fft plot
    def _create_fft_plot(self):
        fft_plot_data = ArrayPlotData(
            index=np.array((0, 1)), yf1=np.array((0, 0)), yf2=np.array((0, 0))
        )
        plot = Plot(fft_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(
            ("index", "yf1"), color=scheme["data 1"], line_width=2, id="0", name="yf1"
        )
        if self.dual:
            plot.plot(
                ("index", "yf2"),
                color=scheme["data 2"],
                line_width=2,
                id="0",
                name="yf2",
            )
        plot.bgcolor = scheme["background"]
        # plot.x_grid = None
        # plot.y_grid = None
        plot.index_axis.title = "frequency [MHz]"
        if self.db:
            plot.value_axis.title = "amplitude [dB]"
        else:
            plot.value_axis.title = "amplitude [a.u.]"
        # plot.tools.append(PanTool(plot))
        # plot.overlays.append(SimpleZoom(plot, enable_wheel=True)) #changed to true
        self.fft_plot_data = fft_plot_data
        self.fft_plot = plot

    @on_trait_change("yf1")
    def _update_fft_plot_value(self):
        y = self.yf1
        n = len(y)
        old_index = self.fft_plot_data.get_data("index")
        if old_index is not None and len(old_index) != n:
            self.fft_plot_data.set_data("index", self.nu)

        if self.db:
            if self.dual:
                zero_db = max(self.yf1.max(), self.yf2.max())
            else:
                zero_db = self.yf1.max()

            spec1 = 20 * np.log10(self.yf1 / zero_db)
        else:
            spec1 = self.yf1
        self.fft_plot_data.set_data("yf1", spec1)

        if self.dual:
            if self.db:
                spec2 = 20 * np.log10(self.yf2 / zero_db)
            else:
                spec2 = self.yf2
            self.fft_plot_data.set_data("yf2", spec2)

    @on_trait_change("db")
    def _update_fft_axis_title(self):
        if self.db:
            self.fft_plot.value_axis.title = "amplitude [dB]"
        else:
            self.fft_plot.value_axis.title = "amplitude [a.u.]"

    # line plot
    def _create_line_plot(self):
        line_plot_data = ArrayPlotData(
            index=np.array((0, 1)), y1=np.array((0, 0)), y2=np.array((0, 0))
        )
        plot = Plot(line_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(
            ("index", "y1"), color=scheme["data 1"], line_width=2, id="0", name="y1"
        )
        if self.dual:
            plot.plot(
                ("index", "y2"),
                color=scheme["data 2"],
                line_width=2,
                id="32",
                name="y2",
            )
        plot.bgcolor = scheme["background"]
        # plot.x_grid = None
        # plot.y_grid = None
        plot.index_axis.title = "tau [ns]"
        plot.value_axis.title = "intensity [a.u.]"
        plot.tools.append(PanTool(plot))
        plot.overlays.append(SimpleZoom(plot, enable_wheel=True))  # changed to true
        self.line_plot_data = line_plot_data
        self.line_plot = plot

    @on_trait_change("y1")
    def _update_line_plot_value(self):
        y = self.y1
        n = len(y)
        old_index = self.line_plot_data.get_data("index")
        if old_index is not None and len(old_index) != n:
            self.line_plot_data.set_data("index", np.arange(n))
        self.line_plot_data.set_data("y1", self.y1)
        if self.dual:
            self.line_plot_data.set_data("y2", self.y2)

    @on_trait_change("count_data")
    def _update_line_plot_index(self):
        self.line_plot_data.set_data("index", self.tau)

    # matrix plot
    def _create_matrix_plot(self):
        matrix_plot_data = ArrayPlotData(image=np.zeros((2, 2)))
        plot = Plot(
            matrix_plot_data,
            width=500,
            height=500,
            resizable="hv",
            padding=8,
            padding_left=48,
            padding_bottom=36,
        )
        plot.index_axis.title = "time [ns]"
        plot.value_axis.title = "data index"
        plot.img_plot(
            "image", xbounds=(0, 1), ybounds=(0, 1), colormap=scheme["colormap"]
        )
        plot.tools.append(PanTool(plot))
        plot.overlays.append(SimpleZoom(plot, enable_wheel=False))
        self.matrix_plot_data = matrix_plot_data
        self.matrix_plot = plot

    @on_trait_change("time_bins, depth")
    def _update_matrix_index(self):
        self.matrix_plot.components[0].index.set_data(
            (self.time_bins[0], self.time_bins[-1]), (0.0, float(self.depth))
        )

    @on_trait_change("count_data")
    def _update_matrix_value(self):
        s = self.count_data.shape
        if not s[0] * s[1] > 1000000:
            self.matrix_plot_data.set_data("image", self.count_data)

    # pulse profile plot
    def _create_pulse_plot(self):
        pulse_plot_data = ArrayPlotData(
            x=np.array((0.0, 0.1, 0.2)), y=np.array((0, 1, 2))
        )
        plot = Plot(pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(
            ("x", "y"), style="line", color=scheme["data 1"], line_width=1, name="data"
        )
        plot.bgcolor = scheme["background"]
        plot.x_grid = None
        plot.y_grid = None
        plot.index_axis.title = "time [ns]"
        plot.value_axis.title = "intensity [a.u.]"
        edge_marker = LinePlot(
            index=ArrayDataSource(np.array((0, 0))),
            value=ArrayDataSource(np.array((0, 1e9))),
            color=scheme["fit 1"],
            index_mapper=LinearMapper(range=plot.index_range),
            value_mapper=LinearMapper(range=plot.value_range),
            name="marker",
        )
        plot.add(edge_marker)
        plot.tools.append(PanTool(plot))
        plot.overlays.append(SimpleZoom(plot, enable_wheel=False))
        self.pulse_plot_data = pulse_plot_data
        self.pulse_plot = plot

    @on_trait_change("time_bins")
    def _update_pulse_index(self):
        self.pulse_plot_data.set_data("x", self.time_bins)

    @on_trait_change("pulse_profile")
    def _update_pulse_value(self):
        self.pulse_plot_data.set_data("y", self.pulse_profile)

    @on_trait_change("flank")
    def _on_flank_change(self, new):
        self.pulse_plot.components[1].index.set_data(np.array((new, new)))

    @on_trait_change("dual")
    def _update_data_depth(self):
        # adjust plot
        plot = self.line_plot
        if self.dual:
            self.line_plot_data.set_data("y2", self.y1)
            plot.plot(
                ("index", "y2"),
                style="line",
                line_width=2,
                color=scheme["data 2"],
                id="32",
                name="y2",
            )
        else:
            for i, c in enumerate(plot.components):
                if c.id == "32":
                    plot.remove(plot.components[i])
        plot.request_redraw()

    # save plots
    def save_line_plot(self, filename):
        # TODO: change color-scheme then save
        self.save_figure(self.line_plot, filename)

    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)

    def save_fft_plot(self, filename):
        self.save_figure(self.fft_plot, filename)

    def save_pulse_plot(self, filename):
        self.save_figure(self.pulse_plot, filename)


class Rabi(Pulsed):
    """Rabi measurement."""

    main_dir = String("\Rabi", desc=".WFM/.PAT folder in AWG", label="Waveform Dir.")
    offset = Range(
        low=-20e6,
        high=20e6,
        value=0.0,
        desc="Offset frequency",
        label="Offset frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    freq_awg = Range(
        low=1,
        high=1e9,
        value=40e6,
        desc="AWG frequency",
        label="AWG frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    freq_tot = Range(
        low=1,
        high=20e9,
        value=2.87e9,
        desc="Total frequency",
        label="Total frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    phase_sin = Range(
        low=-100.0,
        high=100.0,
        value=0.0,
        desc="Multiple of pi",
        label="Phase",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )

    amp1 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor",
        label="wfm amp",
        auto_set=False,
        enter_set=True,
    )

    show_fit = Bool(False, label="show fit")

    # fit parameters
    fit_parameters = np.ones(3)  # changed from 4 to 3
    rabi_period = Float(value=0.0, label="T")
    rabi_frequency = Float(value=0.0, label="f")
    rabi_contrast = Float(value=0.0, label="I")
    rabi_offset = Float(value=0.0, label="t_0")
    pi2 = Float(value=0.0, label="pi/2")
    pi = Float(value=0.0, label="pi")
    pi32 = Float(value=0.0, label="3pi/2")

    get_set_items = Pulsed.get_set_items + [
        "freq_tot",
        "freq_awg",
        "offset",
        "amp1",
        "show_fit",
        "fit_parameters",
        "rabi_frequency",
        "rabi_period",
        "rabi_offset",
        "rabi_contrast",
        "pi2",
        "pi",
        "pi32",
        "phase_sin",
    ]

    def __init__(self, pulse_generator, time_tagger, microwave, awg, **kwargs):
        self.dual = False
        super(Rabi, self).__init__(
            pulse_generator, time_tagger, microwave, awg, **kwargs
        )
        self.show_fit = True
        self.freq_tot = self.freq + self.freq_awg + self.offset

    # Sequence ############################
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        trig_interval = self.trig_interval
        trig_delay = self.trig_delay

        sequence = [(["aom"], laser), ([], wait)]
        for t in tau:
            sequence.append((["awg"], trig_interval))
            sequence.append(([], trig_delay + t))
            # Add switch at the end
            # sequence.append((['mw_y'], trig_delay + t))
            sequence.append((["aom", "laser"], laser))
            sequence.append(([], wait))
        sequence.append((["sequence"], 100))
        return sequence

    def compile_waveforms(self):
        # awg frequency in terms of sampling rate
        # f1 = (self.freq1 - self.freq)/self.sampling
        f1 = self.freq_awg / self.sampling
        phase = self.phase_sin * np.pi

        # pulse objects
        mw = Sin(0, freq=f1, amp=self.amp1, phase=phase)
        zero = Idle(1)

        # pulse sequence
        pulse_sequence = [zero, mw, zero]

        seq = Sequence("RABI")
        for i, t in enumerate(self.tau):
            # increment microwave duration
            mw.duration = t
            name = "RABI_%03i" % i
            waves = IQWaveform(name, pulse_sequence, file_type=0)
            self.waves.append(waves[0])
            self.waves.append(waves[1])
            seq.append(waves, wait=True)
        self.waves.append(seq)
        self.main_wave = seq.name

    def start_up(self):
        """Override for additional stuff to be executed at start up."""
        self.pulse_generator.Night()
        self.microwave.setOutput(self.power, self.freq + self.offset)
        self.awg.set_output(0b11)
        self.awg.run()

    # Plot and Fit ########################
    @on_trait_change("freq, offset, freq_awg")
    def _update_freq_tot(self):
        self.freq_tot = self.freq + self.offset + self.freq_awg

    @on_trait_change("y1")
    def _update_line_plot_value(self):
        """override"""
        y = self.y1
        n = len(y)
        old_index = self.line_plot_data.get_data("index")
        if old_index is not None and len(old_index) != n:
            self.line_plot_data.set_data("index", np.arange(n))
        self.line_plot_data.set_data("y1", y)
        if self.show_fit:
            self.line_plot_data.set_data(
                "fit", fitting.Cosinus(*self.fit_parameters)(self.tau)
            )

    @on_trait_change("show_fit")
    def _plot_fit(self):
        plot = self.line_plot
        if self.show_fit == False:
            while len(plot.components) > 1:
                plot.remove(plot.components[-1])
        else:
            self.line_plot_data.set_data(
                "fit", fitting.Cosinus(*self.fit_parameters)(self.tau)
            )
            plot.plot(
                ("index", "fit"), style="line", line_width=2, color=scheme["fit 1"]
            )
        plot.request_redraw()

    @on_trait_change("y1, show_fit")
    def _update_fit_parameters(self):
        if self.y1 is None:
            return
        else:
            y_offset = self.y1.mean()

            x = self.tau
            y = self.y1 - y_offset

            try:
                # print(fitting.CosinusNoOffsetEstimator)
                # amp, period, phi = fitting.fit(x, y, fitting.CosinusNoOffset, fitting.CosinusNoOffsetEstimator)
                fit_result = fitting.fit_rabi(x, y, np.sqrt(self.y1))
                p, v, q, chisqr = fit_result
                amp, period, phi = p
            except:
                return

            self.fit_parameters = (amp, period, y_offset + phi)
            self.rabi_period = period
            self.rabi_frequency = 1000.0 / period
            self.rabi_contrast = 200 * amp / y_offset
            self.rabi_offset = phi
            self.pi2 = 0.25 * period + phi
            self.pi = 0.5 * period + phi
            self.pi32 = 0.75 * period + phi

    traits_view = View(
        VGroup(
            HGroup(
                Item("submit_button", show_label=False),
                Item("remove_button", show_label=False),
                Item("resubmit_button", show_label=False),
                Item("priority"),
                Item("state", style="readonly"),
                Item("run_time", style="readonly", format_str="%i"),
                Item("stop_time"),
            ),
            Tabbed(
                VGroup(
                    HGroup(
                        HGroup(
                            Item(
                                "freq",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item("offset", enabled_when='state == "idle"'),
                            Item("power", width=-40),
                            label="SMIQ",
                            show_border=True,
                        ),
                        HGroup(
                            Item("reload_awg"),
                            Item("wfm_button", show_label=False),
                            Item("upload_progress", style="readonly", format_str="%i"),
                            Item("main_dir", style="readonly"),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item(
                            "freq_tot",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.6f GHz" % (x * 1e-9),
                            ),
                        ),
                        Item(
                            "freq_awg",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        Item(
                            "phase_sin",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%.2f" % x,
                            ),
                            width=-80,
                        ),
                        Item(
                            "amp1",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%.2f" % x,
                            ),
                            width=-80,
                        ),
                        Item("start_time", width=-40),
                        Item("end_time", width=-40),
                        Item("time_step", width=-40),
                        label="Waveform",
                        show_border=True,
                    ),
                    HGroup(
                        Item("show_fit"),
                        Item(
                            "rabi_contrast",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.1f% %" % x,
                            ),
                        ),
                        Item(
                            "rabi_frequency",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.2f MHz" % x,
                            ),
                        ),
                        Item(
                            "rabi_period",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.2f ns" % x,
                            ),
                        ),
                        Item(
                            "rabi_offset",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.2f ns" % x,
                            ),
                        ),
                        Item(
                            "pi2",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.2f ns" % x,
                            ),
                        ),
                        Item(
                            "pi",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.2f ns" % x,
                            ),
                        ),
                        Item(
                            "pi32",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.2f ns" % x,
                            ),
                        ),
                        label="Fit",
                        show_border=True,
                    ),
                    VGroup(
                        Item(
                            "line_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                        HGroup(
                            Item(
                                "fft_plot",
                                editor=ComponentEditor(),
                                show_label=False,
                                width=500,
                                height=300,
                                resizable=True,
                            ),
                            Item(
                                "matrix_plot",
                                editor=ComponentEditor(),
                                show_label=False,
                                width=500,
                                height=300,
                                resizable=True,
                            ),
                        ),
                    ),
                    label="data",
                ),
                VGroup(
                    HGroup(
                        HGroup(
                            Item("laser", width=-50),
                            Item("wait", width=-50),
                            label="Sequence",
                            show_border=True,
                        ),
                        HGroup(
                            Item("vpp1", width=-50),
                            Item("vpp2", width=-50),
                            Item("sampling", width=-80),
                            Item("trig_interval", width=-80),
                            Item("trig_delay", width=-80),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item("record_length"),
                        Item("bin_width"),
                        Item("time_window_width"),
                        Item("time_window_offset_signal"),
                        Item("time_window_offset_normalize"),
                        Item("dual", enabled_when='state != "run"'),
                        label="Analysis",
                        show_border=True,
                    ),
                    Item(
                        "pulse_plot",
                        editor=ComponentEditor(),
                        show_label=False,
                        width=500,
                        height=300,
                        resizable=True,
                    ),
                    label="settings",
                ),
            ),
        ),
        title="Rabi (AWG)",
        buttons=[],
        resizable=True,
        width=-900,
        height=-800,
        handler=PulsedToolHandler,
        menubar=MenuBar(
            Menu(
                Action(action="load", name="Load"),
                Action(action="save", name="Save (.pyd or .pys)"),
                Action(action="saveMatrixPlot", name="SaveMatrixPlot (.png)"),
                Action(action="saveColorPlot", name="SavePlot (.png)"),
                Action(action="saveAll", name="Save All (.png+.pys)"),
                Action(action="_on_close", name="Quit"),
                name="File",
            ),
        ),
    )


class Trigger(Pulsed):
    """Rabi measurement."""

    SMIQ = Bool(False, label="Turn on SMIQ")
    IQ = Bool(False, label="Use IQ Mixing")
    MW_X = Bool(False, label="Switch on mw_x")

    freq_awg = Range(
        low=1,
        high=1e9,
        value=40e6,
        desc="AWG frequency",
        label="AWG frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    freq_tot = Range(
        low=1,
        high=20e9,
        value=2.87e9,
        desc="Total frequency",
        label="Total frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    phase_sin = Range(
        low=-100.0,
        high=100.0,
        value=0.0,
        desc="Multiple of pi",
        label="Phase",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )

    head = Range(
        low=1.0,
        high=1e7,
        value=15,
        desc="Length of the head [ns]",
        label="Head [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    amp_body = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Scale of the body",
        label="Body amp",
        auto_set=False,
        enter_set=True,
    )
    amp1 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor",
        label="wfm amp",
        auto_set=False,
        enter_set=True,
    )
    n_rep = Range(
        low=0.0,
        high=10000,
        value=40,
        desc="Repetition",
        label="Repetition",
        auto_set=False,
        enter_set=True,
    )

    show_fit = Bool(False, label="show fit")

    # fit parameters
    fit_parameters = np.ones(3)  # changed from 4 to 3
    rabi_period = Float(value=0.0, label="T")
    rabi_frequency = Float(value=0.0, label="f")
    rabi_contrast = Float(value=0.0, label="I")
    rabi_offset = Float(value=0.0, label="t_0")
    pi2 = Float(value=0.0, label="pi/2")
    pi = Float(value=0.0, label="pi")
    pi32 = Float(value=0.0, label="3pi/2")

    get_set_items = Pulsed.get_set_items + [
        "freq_tot",
        "freq_awg",
        "amp1",
        "amp_body",
        "head",
        "SMIQ",
        "MW_X",
        "phase_sin",
        "show_fit",
        "fit_parameters",
        "rabi_frequency",
        "rabi_period",
        "IQ",
        "n_rep",
    ]

    def __init__(self, pulse_generator, time_tagger, microwave, awg, **kwargs):
        self.dual = False
        super(Trigger, self).__init__(
            pulse_generator, time_tagger, microwave, awg, **kwargs
        )
        self.show_fit = True
        self.freq_tot = self.freq + self.freq_awg

    def start_up(self):
        """Override for additional stuff to be executed at start up."""
        self.pulse_generator.Night()
        if self.SMIQ:
            self.microwave.setOutput(self.power, self.freq)
        else:
            self.microwave.setOutput(None, self.freq)
        self.awg.set_output(0b11)
        self.awg.run()

    def shut_down(self):
        """Override for additional stuff to be executed at shut down."""
        self.pulse_generator.Light()
        self.microwave.setOutput(None, self.freq)
        self.awg.stop()
        self.awg.set_output(0b00)

    # Sequence ############################
    def generate_sequence(self):
        t = self.tau[0]
        self.laser
        wait = self.wait
        trig_interval = self.trig_interval

        sequence = [(["green"], wait)]
        if self.MW_X:
            sequence.append(
                (
                    [
                        "green",
                        "blue",
                    ],
                    trig_interval,
                )
            )
            sequence.append(
                (
                    [
                        "green",
                        "mw_x",
                    ],
                    t + trig_interval,
                )
            )
        else:
            sequence.append(
                (
                    [
                        "green",
                        "awg",
                        "blue",
                    ],
                    trig_interval,
                )
            )
            sequence.append(
                (
                    [
                        "green",
                    ],
                    t + trig_interval,
                )
            )

        if self.n_rep > 1:
            return sequence * self.n_rep
        else:
            return sequence

    def compile_waveforms(self):
        # awg frequency in terms of sampling rate
        # f1 = (self.freq1 - self.freq)/self.sampling
        f1 = self.freq_awg / self.sampling
        phase = self.phase_sin * np.pi
        t = self.tau[0]

        if t > self.head:
            t_h = self.head
        else:
            t_h = 0

        # pulse objects
        mw11 = Sin(t_h, freq=f1, amp=self.amp1, phase=phase)
        mw12 = Sin(t - t_h, freq=f1, amp=self.amp_body, phase=phase)
        if self.IQ:
            mw21 = Sin(t_h, freq=f1, amp=self.amp1, phase=phase)
            mw22 = Sin(t - t_h, freq=f1, amp=self.amp_body, phase=phase)
        else:
            mw21 = Sin(t_h, freq=f1, amp=self.amp1, phase=phase - np.pi / 2)
            mw22 = Sin(t - t_h, freq=f1, amp=self.amp_body, phase=phase - np.pi / 2)
        # pulse sequence
        zero = Idle(1)
        pulse_sequence = [zero, mw11, mw12, zero]
        pulse_sequence2 = [zero, mw21, mw22, zero]

        seq = Sequence("TRIG")
        name = "Trig"
        waves = IQWaveform(name, pulse_sequence, file_type=0)
        self.waves.append(waves[0])
        waves = IQWaveform(name, pulse_sequence2, file_type=0)
        self.waves.append(waves[1])
        seq.append(waves, wait=True)
        self.waves.append(seq)
        self.main_wave = seq.name

    # Plot and Fit ########################
    @on_trait_change("freq, freq_awg")
    def _update_freq_tot(self):
        self.freq_tot = self.freq + self.freq_awg

    @on_trait_change("y1")
    def _update_line_plot_value(self):
        """override"""
        y = self.y1
        n = len(y)
        old_index = self.line_plot_data.get_data("index")
        if old_index is not None and len(old_index) != n:
            self.line_plot_data.set_data("index", np.arange(n))
        self.line_plot_data.set_data("y1", y)
        if self.show_fit:
            self.line_plot_data.set_data(
                "fit", fitting.Cosinus(*self.fit_parameters)(self.tau)
            )

    @on_trait_change("show_fit")
    def _plot_fit(self):
        plot = self.line_plot
        if self.show_fit == False:
            while len(plot.components) > 1:
                plot.remove(plot.components[-1])
        else:
            self.line_plot_data.set_data(
                "fit", fitting.Cosinus(*self.fit_parameters)(self.tau)
            )
            plot.plot(
                ("index", "fit"), style="line", line_width=2, color=scheme["fit 1"]
            )
        plot.request_redraw()

    @on_trait_change("y1, show_fit")
    def _update_fit_parameters(self):
        if self.y1 is None:
            return
        else:
            y_offset = self.y1.mean()

            x = self.tau
            y = self.y1 - y_offset

            try:
                # print(fitting.CosinusNoOffsetEstimator)
                # amp, period, phi = fitting.fit(x, y, fitting.CosinusNoOffset, fitting.CosinusNoOffsetEstimator)
                fit_result = fitting.fit_rabi(x, y, np.sqrt(self.y1))
                p, v, q, chisqr = fit_result
                amp, period, phi = p
            except:
                return
            """
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
            
            """

            self.fit_parameters = (amp, period, y_offset + phi)
            self.rabi_period = period
            self.rabi_frequency = 1000.0 / period
            self.rabi_contrast = 200 * amp / y_offset
            self.rabi_offset = phi
            self.pi2 = 0.25 * period + phi
            self.pi = 0.5 * period + phi
            self.pi32 = 0.75 * period + phi

    traits_view = View(
        VGroup(
            HGroup(
                Item("submit_button", show_label=False),
                Item("remove_button", show_label=False),
                Item("resubmit_button", show_label=False),
                Item("priority"),
                Item("state", style="readonly"),
                Item("run_time", style="readonly", format_str="%i"),
                Item("stop_time"),
            ),
            Tabbed(
                VGroup(
                    HGroup(
                        HGroup(
                            Item("MW_X", enabled_when='state != "run"'),
                            Item("SMIQ", enabled_when='state != "run"'),
                            Item(
                                "freq",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item("power", width=-40, enabled_when='state != "run"'),
                            label="SMIQ",
                            show_border=True,
                        ),
                        HGroup(
                            Item("reload_awg"),
                            Item("wfm_button", show_label=False),
                            Item("upload_progress", style="readonly", format_str="%i"),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item("IQ", enabled_when='state != "run"'),
                        Item(
                            "freq_tot",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.4f GHz" % (x * 1e-9),
                            ),
                        ),
                        Item(
                            "freq_awg",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-100,
                        ),
                        Item(
                            "phase_sin",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%.2f" % x,
                            ),
                            width=-40,
                        ),
                        Item(
                            "amp1",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%.2f" % x,
                            ),
                            width=-40,
                        ),
                        Item(
                            "amp_body",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%.2f" % x,
                            ),
                            width=-40,
                        ),
                        label="Waveform",
                        show_border=True,
                    ),
                    HGroup(
                        Item("wait", width=-60, enabled_when='state != "run"'),
                        Item("head", width=-40),
                        Item("start_time", width=-40),
                        Item("end_time", width=-40),
                        Item("time_step", width=-40),
                        label="Time",
                        show_border=True,
                    ),
                    HGroup(
                        Item("vpp1", width=-50),
                        Item("vpp2", width=-50),
                        Item(
                            "sampling",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        Item("trig_interval", width=-40),
                        Item(
                            "n_rep",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%i" % x,
                            ),
                            width=-40,
                        ),
                        label="AWG",
                        show_border=True,
                    ),
                ),
            ),
        ),
        title="Trigger Interval (AWG)",
        buttons=[],
        resizable=True,
        width=-900,
        height=-800,
        handler=PulsedToolHandler,
        menubar=MenuBar(
            Menu(
                Action(action="load", name="Load"),
                Action(action="save", name="Save (.pyd or .pys)"),
                Action(action="saveMatrixPlot", name="SaveMatrixPlot (.png)"),
                Action(action="saveColorPlot", name="SavePlot (.png)"),
                Action(action="saveAll", name="Save All (.png+.pys)"),
                Action(action="_on_close", name="Quit"),
                name="File",
            ),
        ),
    )


class CPMG(Pulsed):
    """CPMG measurement."""

    main_dir = String("\CPMG", desc=".WFM/.PAT folder in AWG", label="Waveform Dir.")
    offset = Range(
        low=-20e6,
        high=20e6,
        value=0.0,
        desc="Offset frequency",
        label="Offset frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    freq_awg = Range(
        low=1,
        high=1e9,
        value=40e6,
        desc="AWG frequency",
        label="AWG frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    freq_tot = Range(
        low=1,
        high=20e9,
        value=2.87e9,
        desc="Total frequency",
        label="Total frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    phase_sin = Range(
        low=-100.0,
        high=100.0,
        value=0.0,
        desc="Multiple of pi",
        label="Phase",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )

    amp1 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor",
        label="wfm amp",
        auto_set=False,
        enter_set=True,
    )

    show_fit = Bool(False, label="show fit")

    t_2pi = Range(
        low=0.0,
        high=2000,
        value=200.0,
        desc="duration of 2pi pulse",
        label="2pi duration",
        auto_set=False,
        enter_set=True,
    )
    n_pi = Range(
        low=1,
        high=100,
        value=4,
        desc="number of pi pulses",
        label="n pi",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    Range(
        low=0,
        high=100,
        value=10,
        desc="Number of Fourier components",
        label="K",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    get_set_items = Pulsed.get_set_items + [
        "freq_tot",
        "freq_awg",
        "offset",
        "amp1",
        "show_fit",
        "t_2pi",
        "n_pi",
        "phase_sin",
    ]

    def __init__(self, pulse_generator, time_tagger, microwave, awg, **kwargs):
        self.dual = False
        super(CPMG, self).__init__(
            pulse_generator, time_tagger, microwave, awg, **kwargs
        )
        self.show_fit = True
        self.freq_tot = self.freq + self.freq_awg + self.offset

    ############################# Sequence ############################
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        trig_interval = self.trig_interval
        trig_delay = self.trig_delay

        t_2pi = self.t_2pi
        t_pi = t_2pi / 2.0
        n_pi = int(self.n_pi)

        sequence = [(["aom"], laser), ([], wait)]

        # Dark
        for t in tau:
            tDD, t_awg = self.get_tau(t, t_2pi, n_pi, state="dark")

            sequence.append((["awg"], trig_interval))
            sequence.append(([], trig_delay + t_awg))
            # Add switch at the end
            # sequence.append((['mw_y'], trig_delay + t_awg))
            sequence.append((["aom", "laser"], laser))
            sequence.append(([], wait))

        # Bright
        for t in tau:
            tDD, t_awg = self.get_tau(t, t_2pi, n_pi, state="bright")

            sequence.append((["awg"], trig_interval))
            sequence.append(([], trig_delay + t_awg))
            # Add switch at the end
            # sequence.append((['mw_y'], trig_delay + t_awg))
            sequence.append((["aom", "laser"], laser))
            sequence.append(([], wait))

        # Bright Reference
        sequence.append((["laser", "green"], laser))
        sequence.append(([], wait))

        # Dark Reference
        sequence.append((["awg"], trig_interval))
        sequence.append(([], trig_delay + t_pi))
        # sequence.append((['mw_y'],trig_delay + t_pi))
        sequence.append((["laser", "green"], laser))
        sequence.append(([], wait))
        sequence.append((["sequence"], 100))
        return sequence

    def compile_waveforms(self):
        # awg frequency in terms of sampling rate
        # f1 = (self.freq1 - self.freq)/self.sampling
        f_res = self.freq_awg / self.sampling
        phase = self.phase_sin * np.pi

        t_2pi = self.t_2pi
        t_pi = t_2pi / 2.0
        t_pi2 = t_2pi / 4.0
        t_3pi2 = t_2pi * 3.0 / 4.0
        n_pi = int(self.n_pi)

        # pulse objects
        mwx_pi = Cos(t_pi, freq=f_res, amp=self.amp1, phase=phase)
        mwy_pi2 = Sin(t_pi2, freq=f_res, amp=self.amp1, phase=phase)
        mwy_3pi2 = Sin(t_3pi2, freq=f_res, amp=self.amp1, phase=phase)
        ref = Sin(t_pi, freq=f_res, amp=self.amp1, phase=phase)
        free = Idle(1)
        zero = Idle(1)

        # pulse sequence
        pulse_dark = [zero, mwy_pi2]
        pulse_dark += [free, mwx_pi, free] * n_pi
        pulse_dark += [mwy_pi2]

        pulse_bright = [zero, mwy_pi2]
        pulse_bright += [free, mwx_pi, free] * n_pi
        pulse_bright += [mwy_3pi2]

        seq = Sequence("CPMG")
        for i, t in enumerate(self.tau):
            # increment microwave duration
            tDD, t_awg = self.get_tau(t, t_2pi, n_pi, state="dark")

            free.duration = tDD
            name = "CPMG_%03i_D" % i
            waves_dark = IQWaveform(name, pulse_dark, file_type=0)
            self.waves.append(waves_dark[0])
            self.waves.append(waves_dark[1])
            seq.append(waves_dark, wait=True)

            1 + t_pi2 * 2 + (t_pi + 2 * tDD) * n_pi

            name = "CPMG_%03i_B" % i
            waves_bright = IQWaveform(name, pulse_bright, file_type=0)
            self.waves.append(waves_bright[0])
            self.waves.append(waves_bright[1])
            seq.append(waves_bright, wait=True)

        pulse_sequencer1 = [zero, ref, zero]
        namer1 = "REF1"
        waver1 = IQWaveform(namer1, pulse_sequencer1, file_type=0)
        self.waves.append(waver1[0])
        self.waves.append(waver1[1])
        seq.append(waver1, wait=True)

        self.waves.append(seq)
        self.main_wave = seq.name

    def get_tau(self, t, t_2pi, n_pi, state="dark"):
        t_pi = t_2pi / 2.0
        t_pi2 = t_2pi / 4.0
        t_3pi2 = t_2pi * 3.0 / 4.0
        tDD = t - t_pi / 2.0
        if state == "dark":
            if tDD < 0:
                return 0.0, t_pi2 + (t_pi) * (n_pi) + t_3pi2
            else:
                # Chose the convention from Tim's paper
                return tDD, t_pi2 + (tDD + t_pi + tDD) * (n_pi) + t_3pi2
        elif state == "bright":
            if tDD < 0:
                return 0.0, t_pi2 + (t_pi) * (n_pi) + t_3pi2
            else:
                # Chose the convention from Tim's paper
                return tDD, t_pi2 + (tDD + t_pi + tDD) * (n_pi) + t_3pi2

    def start_up(self):
        """Override for additional stuff to be executed at start up."""
        self.pulse_generator.Night()
        self.microwave.setOutput(self.power, self.freq + self.offset)
        self.awg.set_output(0b11)
        self.awg.run()

    # Plot and Fit ########################
    @on_trait_change("freq, offset, freq_awg")
    def _update_freq_tot(self):
        self.freq_tot = self.freq + self.offset + self.freq_awg

    def _create_line_plot(self):
        Y = {
            "index": np.array((0, 1)),
            "yd": np.array((0, 0)),
            "yb": np.array((0, 0)),
            "bright": np.array((1, 1)),
            "dark": np.array((0, 0)),
        }
        line_plot_data = ArrayPlotData(**Y)  # Unpack all the items in Y as kwargs

        plot = Plot(line_plot_data, padding=8, padding_left=64, padding_bottom=36)

        plot.plot(("index", "yd"), color="auto", name="yd")
        plot.plot(("index", "yb"), color="auto", name="yb")
        plot.plot(("index", "bright"), color="red", line_width=2, name="bright")
        plot.plot(("index", "dark"), color="black", line_width=2, name="dark")

        plot.bgcolor = scheme["background"]
        plot.x_grid = None
        plot.y_grid = None
        plot.index_axis.title = "tau [ns]"
        plot.value_axis.title = "intensity [a.u.]"
        self.line_plot_data = line_plot_data
        self.line_plot = plot

    @on_trait_change("y1")
    def _update_line_plot_value(self):
        """override"""
        y = self.y1
        # Number of data point except the references
        n_data = len(y) - 2
        n_index = int(n_data // 2)
        ref = y[-2:]
        bright = ref.max()
        dark = ref.min()
        old_index = self.line_plot_data.get_data("index")
        if old_index is not None and len(old_index) != n_data:
            self.line_plot_data.set_data("index", self.tau)
        self.line_plot_data.set_data("yd", y[:n_index])
        self.line_plot_data.set_data("yb", y[n_index : n_index * 2])
        self.line_plot_data.set_data("bright", bright * np.ones(n_index))
        self.line_plot_data.set_data("dark", dark * np.ones(n_index))

    traits_view = View(
        VGroup(
            HGroup(
                Item("submit_button", show_label=False),
                Item("remove_button", show_label=False),
                Item("resubmit_button", show_label=False),
                Item("priority"),
                Item("state", style="readonly"),
                Item("run_time", style="readonly", format_str="%i"),
                Item("stop_time"),
            ),
            Tabbed(
                VGroup(
                    HGroup(
                        HGroup(
                            Item(
                                "freq",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item("offset", enabled_when='state == "idle"'),
                            Item("power", width=-40),
                            label="SMIQ",
                            show_border=True,
                        ),
                        HGroup(
                            Item("reload_awg"),
                            Item("wfm_button", show_label=False),
                            Item("upload_progress", style="readonly", format_str="%i"),
                            Item("main_dir", style="readonly"),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    VGroup(
                        HGroup(
                            Item(
                                "freq_tot",
                                style="readonly",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: " %.6f GHz" % (x * 1e-9),
                                ),
                            ),
                            Item(
                                "freq_awg",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item(
                                "phase_sin",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%.2f" % x,
                                ),
                                width=-80,
                            ),
                            Item(
                                "amp1",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%.2f" % x,
                                ),
                                width=-80,
                            ),
                            label="Waveform",
                            show_border=True,
                        ),
                        HGroup(
                            Item(
                                "t_2pi",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%.2f" % x,
                                ),
                                width=-80,
                            ),
                            Item("n_pi", enabled_when='state == "idle"', width=-80),
                            Item("start_time", width=-40),
                            Item("end_time", width=-40),
                            Item("time_step", width=-40),
                            label="Tau",
                            show_border=True,
                        ),
                        Item(
                            "line_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                    ),
                    label="data",
                ),
                VGroup(
                    HGroup(
                        HGroup(
                            Item("laser", width=-50),
                            Item("wait", width=-50),
                            label="Sequence",
                            show_border=True,
                        ),
                        HGroup(
                            Item("vpp1", width=-50),
                            Item("vpp2", width=-50),
                            Item("sampling", width=-80),
                            Item("trig_interval", width=-80),
                            Item("trig_delay", width=-80),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item("record_length"),
                        Item("bin_width"),
                        Item("time_window_width"),
                        Item("time_window_offset_signal"),
                        Item("time_window_offset_normalize"),
                        Item("dual", enabled_when='state != "run"'),
                        label="Analysis",
                        show_border=True,
                    ),
                    Item(
                        "pulse_plot",
                        editor=ComponentEditor(),
                        show_label=False,
                        width=500,
                        height=300,
                        resizable=True,
                    ),
                    label="settings",
                ),
            ),
        ),
        title="CPMG (AWG)",
        buttons=[],
        resizable=True,
        width=-900,
        height=-800,
        handler=PulsedToolHandler,
        menubar=MenuBar(
            Menu(
                Action(action="load", name="Load"),
                Action(action="save", name="Save (.pyd or .pys)"),
                Action(action="saveMatrixPlot", name="SaveMatrixPlot (.png)"),
                Action(action="saveColorPlot", name="SavePlot (.png)"),
                Action(action="saveAll", name="Save All (.png+.pys)"),
                Action(action="_on_close", name="Quit"),
                name="File",
            ),
        ),
    )


class FID(Pulsed):

    freq1 = Range(
        low=1,
        high=20e9,
        value=2.80e9,
        desc="microwave frequency transition 1",
        label="frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    freq2 = Range(
        low=1,
        high=20e9,
        value=2.95e9,
        desc="microwave frequency transition 2",
        label="frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    amp1 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor transition 1",
        label="wfm amp",
        auto_set=False,
        enter_set=True,
    )
    amp2 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor transition 2",
        label="wfm amp",
        auto_set=False,
        enter_set=True,
    )

    pi2_1 = Range(
        low=1.0,
        high=100000.0,
        value=30,
        desc="length of  pi/2 pulse transition 1 [ns]",
        label="pi/2 [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    pi32_1 = Range(
        low=1.0,
        high=100000.0,
        value=90,
        desc="length of 3pi/2 pulse transition 1 [ns]",
        label="3pi/2 [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    pi_2 = Range(
        low=1.0,
        high=100000.0,
        value=60,
        desc="length of pi pulse transition 2[ns]",
        label="pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    dq = Bool(True, label="DQ", desc="create double quantum coherence")

    get_set_items = Pulsed.get_set_items + [
        "freq1",
        "freq2",
        "amp1",
        "pi2_1",
        "pi32_1",
        "pi_2",
        "dq",
    ]

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t0 = 2 * self.pi2_1 + 200 + 2 * self.dq * self.pi_2
        sequence = [(["aom"], laser), ([], wait)]
        for t in tau:
            sequence.append((["awg"], 200))
            sequence.append(([], t + t0))
            sequence.append((["aom", "laser"], laser))
            sequence.append(([], wait))
        if self.dual:
            t0 += self.pi32_1 - self.pi2_1
            for t in tau:
                sequence.append((["awg"], 200))
                sequence.append(([], t + t0))
                sequence.append((["aom", "laser"], laser))
                sequence.append(([], wait))

        sequence.append((["sequence"], 100))
        return sequence

    def compile_waveforms(self):
        # frequencies in terms of sampling rate
        f1 = (self.freq1 - self.freq) / self.sampling
        f2 = (self.freq2 - self.freq) / self.sampling

        tau = self.tau * self.sampling / 1e9

        # pulse durations
        d = {}
        d["pi/2 +"] = self.pi2_1 * self.sampling / 1e9
        d["3pi/2 +"] = self.pi32_1 * self.sampling / 1e9
        d["pi -"] = self.pi_2 * self.sampling / 1e9

        # pulse objects
        p = {}
        p["pi/2 +"] = Sin(d["pi/2 +"], freq=f1, amp=self.amp1, phase=0)
        p["3pi/2 +"] = Sin(d["3pi/2 +"], freq=f1, amp=self.amp1, phase=0)
        p["pi -"] = Sin(d["pi -"], freq=f2, amp=self.amp2, phase=0)

        carry = Idle(0)
        zero = Idle(1)

        # pulse sequences
        if self.dq:
            sup_seq = [
                zero,
                p["pi/2 +"],
                p["pi -"],
            ]
            map_seq = [
                carry,
                p["pi -"],
                p["pi/2 +"],
                zero,
            ]
            m32_seq = [
                carry,
                p["pi -"],
                p["3pi/2 +"],
                zero,
            ]
        else:
            sup_seq = [
                zero,
                p["pi/2 +"],
            ]
            map_seq = [
                carry,
                p["pi/2 +"],
                zero,
            ]
            m32_seq = [
                carry,
                p["3pi/2 +"],
                zero,
            ]

        # main sequence
        sequences = []
        main_seq = Sequence("FID.SEQ")

        # waveforms
        waves = []
        idle = Waveform("IDLE", Idle(256))
        waves.append(idle)

        # superposition pulse
        begin_wfm = IQWaveform("FID_A", sup_seq)
        waves.append(begin_wfm["I"])
        waves.append(begin_wfm["Q"])

        t_b, stub = begin_wfm["I"].duration, begin_wfm["I"].stub

        for i, t in enumerate(tau):
            if t < stub:
                # substitute sub_seq with a single waveform
                carry.duration = t
                sub_wfm = IQWaveform("FID_SUB_%03i" % i, sup_seq + map_seq)
                waves.append(sub_wfm["I"])
                waves.append(sub_wfm["Q"])
                main_seq.append(sub_wfm, wait=True)
            else:
                # sub sequence
                sub_seq = Sequence("FID_SUB_%03i" % i)
                # Prepare emitter and superimpose sensor
                sub_seq.append(begin_wfm)

                # free evolution
                n, carry.duration = int(t - stub) / 256, int(t - stub) % 256
                t_offset = t_b + n * 256

                if n > 0:
                    sub_seq.append(idle, idle, repeat=n)

                # mapping pulse
                end_wfm = IQWaveform("FID_B_%03i" % i, map_seq, t_offset)
                waves.append(end_wfm["I"])
                waves.append(end_wfm["Q"])
                sub_seq.append(end_wfm)

                sequences.append(sub_seq)
                main_seq.append(sub_seq, wait=True)

        # Additional Waveforms for 3pi/2 data set
        if self.dual:
            for i, t in enumerate(tau):
                i += len(tau)
                if t < stub:
                    # substitute sub_seq with a single waveform
                    carry.duration = t
                    sub_wfm = IQWaveform("FID_SUB_%03i" % i, sup_seq + map_seq)
                    waves.append(sub_wfm["I"])
                    waves.append(sub_wfm["Q"])
                    main_seq.append(sub_wfm, wait=True)
                else:
                    # sub sequence
                    sub_seq = Sequence("FID_SUB_%03i" % i)
                    sub_seq.append(begin_wfm)

                    # free evolution
                    n, carry.duration = int(t - stub) / 256, int(t - stub) % 256
                    t_offset = t_b + n * 256

                    if n > 0:
                        sub_seq.append(idle, idle, repeat=n)

                    # mapping pulse
                    end_wfm = IQWaveform("FID_C_%03i" % i, m32_seq, t_offset)
                    waves.append(end_wfm["I"])
                    waves.append(end_wfm["Q"])
                    sub_seq.append(end_wfm)

                    sequences.append(sub_seq)
                    main_seq.append(sub_seq, wait=True)

        sequences.append(main_seq)
        self.main_wave = main_seq.name

        # store waves and sequences
        self.waves = waves + sequences

    traits_view = View(
        VGroup(
            HGroup(
                Item("submit_button", show_label=False),
                Item("remove_button", show_label=False),
                Item("priority"),
                Item("state", style="readonly"),
                Item("run_time", style="readonly", format_str="%i"),
                Item("stop_time"),
            ),
            Tabbed(
                VGroup(
                    HGroup(
                        HGroup(
                            Item(
                                "freq",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item("power", width=-40),
                            label="SMIQ",
                            show_border=True,
                        ),
                        HGroup(
                            Item("reload_awg"),
                            Item("wfm_button", show_label=False),
                            Item("upload_progress", style="readonly", format_str="%i"),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item(
                            "freq1",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        Item("amp1", width=-40),
                        Item("pi2_1", width=-40),
                        Item("pi32_1", width=-40),
                        label="Main Transition",
                        show_border=True,
                    ),
                    HGroup(
                        Item(
                            "freq2",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        Item("amp2", width=-40),
                        Item("pi_2", width=-40),
                        Item("dq"),
                        label="Secondary Transition",
                        show_border=True,
                    ),
                    HGroup(
                        Item("start_time", width=-40),
                        Item("end_time", width=-40),
                        Item("time_step", width=-40),
                        label="Sweep",
                        show_border=True,
                    ),
                    VSplit(
                        Item(
                            "line_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                        Item(
                            "fft_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                        Item(
                            "matrix_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                    ),
                    label="data",
                ),
                VGroup(
                    HGroup(
                        HGroup(
                            Item("laser", width=-50),
                            Item("wait", width=-50),
                            label="Sequence",
                            show_border=True,
                        ),
                        HGroup(
                            Item("vpp1", width=-50),
                            Item("vpp2", width=-50),
                            Item("sampling", width=-80),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item("record_length"),
                        Item("bin_width"),
                        Item("time_window_width"),
                        Item("time_window_offset_signal"),
                        Item("time_window_offset_normalize"),
                        Item("dual", enabled_when='state != "run"'),
                        label="Analysis",
                        show_border=True,
                    ),
                    Item(
                        "pulse_plot",
                        editor=ComponentEditor(),
                        show_label=False,
                        width=500,
                        height=300,
                        resizable=True,
                    ),
                    label="settings",
                ),
            ),
        ),
        title="FID (AWG)",
        buttons=[],
        resizable=True,
        width=-900,
        height=-800,
        handler=PulsedToolHandler,
    )


class Hahn(Pulsed):

    freq1 = Range(
        low=1,
        high=20e9,
        value=2.80e9,
        desc="microwave frequency transition 1",
        label="1: frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    freq2 = Range(
        low=1,
        high=20e9,
        value=2.95e9,
        desc="microwave frequency transition 2",
        label="2: frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    amp1 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor transition 1",
        label="1: wfm amp",
    )
    amp2 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor transition 2",
        label="2: wfm amp",
    )

    pi2_1 = Range(
        low=1.0,
        high=100000.0,
        value=30,
        desc="length of pi/2 pulse transition 1 [ns]",
        label="1: pi/2 [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    pi_1 = Range(
        low=1.0,
        high=100000.0,
        value=60,
        desc="length of pi pulse transition 1 [ns]",
        label="1: pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    pi32_1 = Range(
        low=1.0,
        high=100000.0,
        value=90,
        desc="length of 3pi/2 pulse transition 1 [ns]",
        label="1: 3pi/2 [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    pi_2 = Range(
        low=1.0,
        high=100000.0,
        value=60,
        desc="length of pi pulse transition 2[ns]",
        label="2: pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    dq = Bool(True, desc="create double quantum coherence")

    get_set_items = Pulsed.get_set_items + [
        "freq1",
        "freq2",
        "amp1",
        "pi2_1",
        "pi_1",
        "pi32_1",
        "pi_2",
        "dq",
    ]

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t0 = 2 * self.pi2_1 + self.pi_1 + 200 + 2 * self.dq * self.pi_2
        sequence = [(["aom"], laser), ([], wait)]
        for t in tau:
            sequence.append((["awg"], 200))
            sequence.append(([], 2 * t + t0))
            sequence.append((["aom", "laser"], laser))
            sequence.append(([], wait))
        if self.dual:
            t0 += self.pi32_1 - self.pi2_1
            for t in tau:
                sequence.append((["awg"], 200))
                sequence.append(([], 2 * t + t0))
                sequence.append((["aom", "laser"], laser))
                sequence.append(([], wait))

        sequence.append((["sequence"], 100))
        return sequence

    def compile_waveforms(self):
        # frequencies in terms of sampling rate
        f1 = (self.freq1 - self.freq) / self.sampling
        f2 = (self.freq2 - self.freq) / self.sampling

        tau = self.tau * self.sampling / 1e9

        # pulse durations
        d = {}
        d["pi/2 +"] = self.pi2_1 * self.sampling / 1e9
        d["pi +"] = self.pi_1 * self.sampling / 1e9
        d["3pi/2 +"] = self.pi32_1 * self.sampling / 1e9
        d["pi -"] = self.pi_2 * self.sampling / 1e9

        # pulse objects
        p = {}
        p["pi/2 +"] = Sin(d["pi/2 +"], freq=f1, amp=self.amp1, phase=0)
        p["pi +"] = Sin(d["pi +"], freq=f1, amp=self.amp1, phase=0)
        p["3pi/2 +"] = Sin(d["3pi/2 +"], freq=f1, amp=self.amp1, phase=0)
        p["pi -"] = Sin(d["pi -"], freq=f2, amp=self.amp2, phase=0)

        carry = Idle(0)
        zero = Idle(1)

        # pulse sequences
        if self.dq:
            sup_seq = [
                zero,
                p["pi/2 +"],
                p["pi -"],
            ]
            ref_seq = [
                carry,
                p["pi -"],
                p["pi +"],
                p["pi -"],
            ]
            map_seq = [
                carry,
                p["pi -"],
                p["pi/2 +"],
                zero,
            ]
            m32_seq = [
                carry,
                p["pi -"],
                p["3pi/2 +"],
                zero,
            ]

        else:
            sup_seq = [
                zero,
                p["pi/2 +"],
            ]
            ref_seq = [
                carry,
                p["pi +"],
            ]
            map_seq = [
                carry,
                p["pi/2 +"],
                zero,
            ]
            m32_seq = [
                carry,
                p["3pi/2 +"],
                zero,
            ]

        # main sequence
        sequences = []
        sequences2 = []
        main_seq = Sequence("HAHN")

        # waveforms
        waves = []
        idle = Waveform("IDLE", Idle(256))
        waves.append(idle)

        # superposition pulse
        begin_wfm = IQWaveform("HAHN_A", sup_seq)
        waves.append(begin_wfm["I"])
        waves.append(begin_wfm["Q"])

        t_a, stub_a = begin_wfm["I"].duration, begin_wfm["I"].stub

        for i, t in enumerate(tau):
            if t < stub_a:
                # substitute sub_seq with a single waveform
                carry.duration = t
                sub_wfm = IQWaveform("HAHN_SUB_%03i" % i, sup_seq + ref_seq + map_seq)
                waves.append(sub_wfm["I"])
                waves.append(sub_wfm["Q"])
                main_seq.append(sub_wfm, wait=True)

                if self.dual:
                    s32_wfm = IQWaveform(
                        "HAHN_SUB_%03i" % (i + len(tau)), sup_seq + ref_seq + m32_seq
                    )
                    waves.append(sub_wfm["I"])
                    waves.append(sub_wfm["Q"])

                    s32_seq = Sequence("HAHN_SUB_%03i" % (i + len(tau)))
                    s32_seq.append(s32_wfm)
                    sequences2.append(s32_seq)

            else:
                # sub sequence
                sub_seq = Sequence("HAHN_SUB_%03i" % i)
                sub_seq.append(begin_wfm)

                # free evolution
                n, carry.duration = int(t - stub_a) / 256, int(t - stub_a) % 256
                t_offset = t_a + n * 256

                if n > 0:
                    sub_seq.append(idle, idle, repeat=n)

                # refocus pulse
                mid_wfm = IQWaveform("HAHN_B_%03i" % i, ref_seq, t_offset)
                waves.append(mid_wfm["I"])
                waves.append(mid_wfm["Q"])
                sub_seq.append(mid_wfm)

                t_b, stub_b = mid_wfm["I"].duration, mid_wfm["I"].stub

                # free evolution
                m, carry.duration = int(t - stub_b) / 256, int(t - stub_b) % 256
                t_offset += t_b + m * 256

                if m > 0:
                    sub_seq.append(idle, idle, repeat=m)

                # mapping pulse
                end_wfm = IQWaveform("HAHN_C_%03i" % i, map_seq, t_offset)
                waves.append(end_wfm["I"])
                waves.append(end_wfm["Q"])
                sub_seq.append(end_wfm)

                sequences.append(sub_seq)
                main_seq.append(sub_seq, wait=True)

                if self.dual:
                    s32_seq = Sequence("HAHN_SUB_%03i" % (i + len(tau)))
                    s32_seq.append(begin_wfm)
                    if n > 0:
                        s32_seq.append(idle, idle, repeat=n)
                    s32_seq.append(mid_wfm)
                    if m > 0:
                        s32_seq.append(idle, idle, repeat=n)
                    # mapping pulse
                    end_wfm = IQWaveform("HAHN_D_%03i" % i, m32_seq, t_offset)
                    waves.append(end_wfm["I"])
                    waves.append(end_wfm["Q"])
                    s32_seq.append(end_wfm)

                    sequences2.append(s32_seq)

        if self.dual:
            for seq in sequences2:
                main_seq.append(seq, wait=True)
            sequences.extend(sequences2)
        sequences.append(main_seq)
        self.main_wave = main_seq.name

        # store waves and sequences
        self.waves = waves + sequences

    traits_view = View(
        VGroup(
            HGroup(
                Item("submit_button", show_label=False),
                Item("remove_button", show_label=False),
                Item("priority"),
                Item("state", style="readonly"),
                Item("run_time", style="readonly", format_str="%i"),
                Item("stop_time"),
            ),
            Tabbed(
                VGroup(
                    HGroup(
                        HGroup(
                            Item(
                                "freq",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item("power", width=-40),
                            label="SMIQ",
                            show_border=True,
                        ),
                        HGroup(
                            Item("reload_awg"),
                            Item("wfm_button", show_label=False),
                            Item("upload_progress", style="readonly", format_str="%i"),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item(
                            "freq1",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        Item("amp1", width=-40),
                        Item("pi2_1", width=-40),
                        Item("pi_1", width=-40),
                        Item("pi32_1", width=-40),
                        label="Main Transition",
                        show_border=True,
                    ),
                    HGroup(
                        Item(
                            "freq2",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        Item("amp2", width=-40),
                        Item("pi_2", width=-40),
                        Item("dq"),
                        label="Secondary Transition",
                        show_border=True,
                    ),
                    HGroup(
                        Item("start_time", width=-40),
                        Item("end_time", width=-40),
                        Item("time_step", width=-40),
                        label="Sweep",
                        show_border=True,
                    ),
                    VSplit(
                        Item(
                            "line_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                        Item(
                            "fft_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                        Item(
                            "matrix_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                    ),
                    label="data",
                ),
                VGroup(
                    HGroup(
                        HGroup(
                            Item("laser", width=-50),
                            Item("wait", width=-50),
                            label="Sequence",
                            show_border=True,
                        ),
                        HGroup(
                            Item("vpp1", width=-50),
                            Item("vpp2", width=-50),
                            Item("sampling", width=-80),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item("record_length"),
                        Item("bin_width"),
                        Item("time_window_width"),
                        Item("time_window_offset_signal"),
                        Item("time_window_offset_normalize"),
                        Item("dual", enabled_when='state != "run"'),
                        label="Analysis",
                        show_border=True,
                    ),
                    Item(
                        "pulse_plot",
                        editor=ComponentEditor(),
                        show_label=False,
                        width=500,
                        height=300,
                        resizable=True,
                    ),
                    label="settings",
                ),
            ),
        ),
        title="Hahn (AWG)",
        buttons=[],
        resizable=True,
        width=-900,
        height=-800,
        handler=PulsedToolHandler,
    )


class DEER(Pulsed):

    freq1 = Range(
        low=1,
        high=20e9,
        value=2.80e9,
        desc="microwave frequency [Hz]",
        label="frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    amp1 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor",
        label="wfm amp",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    pi2_1 = Range(
        low=1.0,
        high=100000.0,
        value=30,
        desc="length of pi/2 pulse[ns]",
        label="pi/2 [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    pi_1 = Range(
        low=1.0,
        high=100000.0,
        value=60,
        desc="length of pi pulse [ns]",
        label="pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    pi32_1 = Range(
        low=1.0,
        high=100000.0,
        value=90,
        desc="length of 3pi/2 pulse [ns]",
        label="3pi/2 [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    freq2 = Range(
        low=1,
        high=20e9,
        value=2.95e9,
        desc="microwave frequency [Hz]",
        label="frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    amp2 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor",
        label="wfm amp",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    pi_2 = Range(
        low=1.0,
        high=100000.0,
        value=60,
        desc="length of pi pulse[ns]",
        label="pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    freq3 = Range(
        low=1,
        high=20e9,
        value=2.95e9,
        desc="microwave frequency [Hz]",
        label="frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    amp3 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor",
        label="wfm amp",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    pi_3 = Range(
        low=1.0,
        high=100000.0,
        value=60,
        desc="length of pi pulse [ns]",
        label="pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    freq4 = Range(
        low=1,
        high=20e9,
        value=2.95e9,
        desc="microwave frequency [Hz]",
        label="frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    amp4 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor",
        label="wfm amp",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    pi_4 = Range(
        low=1.0,
        high=100000.0,
        value=60,
        desc="length of pi pulse [ns]",
        label="pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    dqa = Bool(True, desc="create double quantum coherence on sensor", label="use")
    dqb = Bool(True, desc="double quantum flip emitter", label="use")

    echo_time = Range(
        low=500,
        high=1e7,
        value=50000,
        desc="Free evolution time of sensor spin [ns]",
        label="echo [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    get_set_items = Pulsed.get_set_items + [
        "freq1",
        "freq2",
        "freq3",
        "freq4",
        "amp1",
        "amp2",
        "amp3",
        "amp4",
        "pi2_1",
        "pi_1",
        "pi32_1",
        "pi_2",
        "pi_3",
        "pi_4",
        "dqa",
        "dqb",
        "echo_time",
    ]

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t0 = 2 * self.pi2_1 + self.pi_1 + self.pi_3 + 200
        t0 += self.dqa * 4 * self.pi_2  # if self.dqa add stuff
        t0 += self.dqb * 2 * self.pi_4
        sequence = [(["aom"], laser), ([], wait)]
        for i in (0, 1):
            for t in tau:
                sequence.append((["awg"], 200))
                sequence.append(([], t + t0))
                sequence.append((["aom", "laser"], laser))
                sequence.append(([], wait))
            if not self.dual:
                break
            t0 += self.pi32_1 - self.pi2_1

        sequence.append((["sequence"], 100))
        return sequence

    def compile_waveforms(self):
        # parameters
        echo_time = self.echo_time * self.sampling / 1e9
        tau = self.tau * self.sampling / 1e9

        # frequencies
        f = {}
        f["A+"] = (self.freq1 - self.freq) / self.sampling
        f["A-"] = (self.freq2 - self.freq) / self.sampling
        f["B+"] = (self.freq3 - self.freq) / self.sampling
        f["B-"] = (self.freq4 - self.freq) / self.sampling

        # pulse durations
        d = {}
        d["pi/2 A+"] = self.pi2_1 * 1e-9 * self.sampling
        d["pi A+"] = self.pi_1 * 1e-9 * self.sampling
        d["3pi/2 A+"] = self.pi32_1 * 1e-9 * self.sampling
        d["pi A-"] = self.pi_2 * 1e-9 * self.sampling
        d["pi B+"] = self.pi_3 * 1e-9 * self.sampling
        d["pi B-"] = self.pi_4 * 1e-9 * self.sampling

        # pulse objects
        p = {}
        p["pi/2 A+"] = Sin(d["pi/2 A+"], freq=f["A+"], amp=self.amp1, phase=0)
        p["pi A+"] = Sin(d["pi A+"], freq=f["A+"], amp=self.amp1, phase=0)
        p["3pi/2 A+"] = Sin(d["3pi/2 A+"], freq=f["A+"], amp=self.amp1, phase=0)
        p["pi A-"] = Sin(d["pi A-"], freq=f["A-"], amp=self.amp2, phase=0)
        p["pi B+"] = Sin(d["pi B+"], freq=f["B+"], amp=self.amp3, phase=0)
        p["pi B-"] = Sin(d["pi B-"], freq=f["B-"], amp=self.amp4, phase=0)

        carry = Idle(0)
        zero = Idle(1)

        # pulse sequences

        # sensor
        if self.dqa:
            sup_seq = [
                p["pi/2 A+"],
                p["pi A-"],
            ]
            ref_seq = [
                carry,
                p["pi A-"],
                p["pi A+"],
                p["pi A-"],
            ]
            map_seq = [
                carry,
                p["pi A-"],
                p["pi/2 A+"],
                zero,
            ]
            m32_seq = [
                carry,
                p["pi A-"],
                p["3pi/2 A+"],
                zero,
            ]
        else:
            sup_seq = [
                p["pi/2 A+"],
            ]
            ref_seq = [
                carry,
                p["pi A+"],
            ]
            map_seq = [
                carry,
                p["pi/2 A+"],
                zero,
            ]
            m32_seq = [
                carry,
                p["3pi/2 A+"],
                zero,
            ]
        # emitter
        if self.dqb:
            pre_seq = [
                zero,
                p["pi B-"],
            ]
            flp_seq = [
                carry,
                p["pi B-"],
                p["pi B+"],
            ]
        else:
            pre_seq = [
                zero,
            ]
            flp_seq = [
                carry,
                p["pi B+"],
            ]

        # main sequence
        sequences = []
        main_seq = Sequence("DEER")

        # waveforms
        waves = []
        idle = Waveform("IDLE", Idle(256))
        waves.append(idle)

        begin_wfm = IQWaveform("DEER_A", pre_seq + sup_seq)
        waves.append(begin_wfm["I"])
        waves.append(begin_wfm["Q"])

        stub = begin_wfm["I"].stub
        n, carry.duration = int(echo_time - stub) / 256, int(echo_time - stub) % 256
        t_offset = begin_wfm["I"].duration + n * 256

        mid_wfm = IQWaveform("DEER_B", ref_seq, t_offset)
        waves.append(mid_wfm["I"])
        waves.append(mid_wfm["Q"])

        stub = mid_wfm["I"].stub

        for i, t in enumerate(tau):
            # sub sequence
            sub_seq = Sequence("DEER_SUB_%03i" % i)
            # Prepare emitter and superimpose sensor
            sub_seq.append(begin_wfm)

            # Free evolution
            if n > 0:
                sub_seq.append(idle, idle, repeat=n)

            # Refocus sensor
            sub_seq.append(mid_wfm)

            # Free evolution
            m, carry.duration = int(t - stub) / 256, int(t - stub) % 256
            if m > 0:
                sub_seq.append(idle, idle, repeat=m)
            t_offset = begin_wfm["I"].duration + mid_wfm["I"].duration + (n + m) * 256

            # Flip emitter
            flp_wfm = IQWaveform("DEER_C_%03i" % i, flp_seq, t_offset)
            waves.append(flp_wfm["I"])
            waves.append(flp_wfm["Q"])
            sub_seq.append(flp_wfm)

            # Free evolution
            rest_evo = int(echo_time - flp_wfm["I"].duration - stub - m * 256)
            l, carry.duration = rest_evo / 256, rest_evo % 256

            if l > 0:
                sub_seq.append(idle, idle, repeat=l)
            t_offset = t_offset + flp_wfm["I"].duration + l * 256

            # Map sensor coherences
            end_wfm = IQWaveform("DEER_D_%03i" % i, map_seq, t_offset)
            waves.append(end_wfm["I"])
            waves.append(end_wfm["Q"])
            sub_seq.append(end_wfm)

            sequences.append(sub_seq)
            main_seq.append(sub_seq, wait=True)
            max_i = i

        if self.dual:
            for i, t in enumerate(tau):
                i += max_i
                # not safe to copy Sequence objects -> compile new one
                # sub sequence
                sub_seq = Sequence("DEER_SUB_%03i" % i)
                # Prepare emitter and superimpose sensor
                sub_seq.append(begin_wfm)

                # Free evolution
                if n > 0:
                    sub_seq.append(idle, idle, repeat=n)

                # Refocus sensor
                sub_seq.append(mid_wfm)

                # Free evolution
                m, carry.duration = int(t - stub) / 256, int(t - stub) % 256
                if m > 0:
                    sub_seq.append(idle, idle, repeat=m)
                t_offset = (
                    begin_wfm["I"].duration + mid_wfm["I"].duration + (n + m) * 256
                )

                # Flip emitter
                flp_wfm = IQWaveform("DEER_E_%03i" % i, flp_seq, t_offset)
                waves.append(flp_wfm["I"])
                waves.append(flp_wfm["Q"])
                sub_seq.append(flp_wfm)

                # Free evolution
                rest_evo = int(echo_time - flp_wfm["I"].duration - stub - m * 256)
                l, carry.duration = rest_evo / 256, rest_evo % 256

                if l > 0:
                    sub_seq.append(idle, idle, repeat=l)
                t_offset = t_offset + flp_wfm["I"].duration + l * 256

                # Map sensor coherences
                end_wfm = IQWaveform("DEER_F_%03i" % i, m32_seq, t_offset)
                waves.append(end_wfm["I"])
                waves.append(end_wfm["Q"])
                sub_seq.append(end_wfm)

                sequences.append(sub_seq)
                main_seq.append(sub_seq, wait=True)

        sequences.append(main_seq)
        self.main_wave = main_seq.name

        # store waves and sequences
        self.waves = waves + sequences

    traits_view = View(
        VGroup(
            HGroup(
                Item("submit_button", show_label=False),
                Item("remove_button", show_label=False),
                Item("priority"),
                Item("state", style="readonly"),
                Item("run_time", style="readonly", format_str="%i"),
                Item("stop_time"),
            ),
            Tabbed(
                VGroup(
                    VSplit(
                        Item(
                            "line_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                        Item(
                            "fft_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                        Item(
                            "matrix_plot",
                            editor=ComponentEditor(),
                            show_label=False,
                            width=500,
                            height=300,
                            resizable=True,
                        ),
                    ),
                    label="data",
                ),
                VGroup(
                    HGroup(
                        HGroup(
                            Item(
                                "freq",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item("power", width=40),
                            label="SMIQ",
                            show_border=True,
                        ),
                        HGroup(
                            Item("reload_awg"),
                            Item("wfm_button", show_label=False),
                            Item("upload_progress", style="readonly", format_str="%i"),
                            Item("vpp1", width=50),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    VGroup(
                        HGroup(
                            Item(
                                "freq1",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item("pi2_1", width=40),
                            Item("pi_1", width=40),
                            Item("pi32_1", width=40),
                            Item("amp1", width=40),
                            label="Transition 1",
                            show_border=True,
                        ),
                        HGroup(
                            Item(
                                "freq2",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item("pi_2", width=40),
                            Item("amp2", width=40),
                            Item("dqa"),
                            label="Transition 2",
                            show_border=True,
                        ),
                        label="Sensor Spin",
                        show_border=True,
                    ),
                    VGroup(
                        HGroup(
                            Item(
                                "freq3",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item("pi_3", width=40),
                            Item("amp3", width=40),
                            label="Transition 1",
                            show_border=True,
                        ),
                        HGroup(
                            Item(
                                "freq4",
                                editor=TextEditor(
                                    auto_set=False,
                                    enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: "%e" % x,
                                ),
                                width=-80,
                            ),
                            Item("pi_4", width=40),
                            Item("amp4", width=40),
                            Item("dqb"),
                            label="Transition 2",
                            show_border=True,
                        ),
                        label="Emitter Spin",
                        show_border=True,
                    ),
                    HGroup(
                        Item("echo_time", width=40),
                        Item("start_time", width=40),
                        Item("end_time", width=40),
                        Item("time_step", width=40),
                        label="Sweep",
                        show_border=True,
                    ),
                    label="parameters",
                ),
                VGroup(
                    HGroup(
                        HGroup(
                            Item("laser", width=-50),
                            Item("wait", width=-50),
                            label="Sequence",
                            show_border=True,
                        ),
                        HGroup(
                            Item("vpp1", width=-50),
                            Item("vpp2", width=-50),
                            Item("sampling", width=-80),
                            label="AWG",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item("record_length"),
                        Item("bin_width"),
                        Item("time_window_width"),
                        Item("time_window_offset_signal"),
                        Item("time_window_offset_normalize"),
                        Item("dual", enabled_when='state != "run"'),
                        label="Analysis",
                        show_border=True,
                    ),
                    Item(
                        "pulse_plot",
                        editor=ComponentEditor(),
                        show_label=False,
                        width=500,
                        height=300,
                        resizable=True,
                    ),
                    label="settings",
                ),
            ),
        ),
        title="DEER",
        buttons=[],
        resizable=True,
        width=-900,
        height=-800,
        handler=PulsedToolHandler,
    )
