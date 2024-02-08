from tokenize import String
import numpy as np

fft = np.fft.fft
ifft = np.fft.ifft


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
from traitsui.file_dialog import save_file

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

    """
    For 2 awg, awg520 and awg610.
    Since awg610 only one channel, so:
    set_vpp() in load_wfm_rf is different
    no vpp2

    
    """

    # awg#
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
        value=1.0,
        label="vpp CH1 [V]",
        desc="Set output voltage peak to peak [V] of CH1.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    vpp2 = Range(
        low=0.0,
        high=2.0,
        value=1.0,
        label="vpp CH2 [V]",
        desc="Set output voltage peak to peak [V] of CH2.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    trig_interval = Range(
        low=11.0,
        high=10000.0,
        value=40.0,
        label="Trig Interval",
        desc="Set the trigger duration for the AWG.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    trig_delay = Range(
        low=11.0,
        high=10000.0,
        value=50.0,
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

    # awg_rf#
    # Directory storing .WFM/.PAT files in AWG
    # Default directory is \waves
    main_dir_rf = String(
        "\waves", desc=".WFM/.PAT folder in AWG", label="Waveform Dir.:", mode="text"
    )
    wfm_button_rf = Button(
        label="Load rf", desc="Compile waveforms and upload them to the AWG rf."
    )

    # Traits ########################
    reload_awg_rf = Bool(
        True, label="reload rf", desc="Compile waveforms upon start up."
    )
    upload_progress_rf = Float(
        label="Upload progress rf", desc="Progress uploading waveforms", mode="text"
    )

    vpp_rf = Range(
        low=0.0,
        high=2.0,
        value=0.2,
        label="vpp CH1 rf [V]",
        desc="Set output voltage peak to peak [V] of CH1.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    trig_interval_rf = Range(
        low=11.0,
        high=10000.0,
        value=40,
        label="Trig Interval rf",
        desc="Set the trigger duration for the AWG.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    trig_delay_rf = Range(
        low=11.0,
        high=10000.0,
        value=440,
        label="Trig Delay rf",
        desc="Trigger delay of the AWG.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    sampling_rf = Range(
        low=50.0e3,
        high=2.6e9,
        value=50e6,
        label="sampling rf [Hz]",
        desc="Set the sampling rate of the AWG.",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    waves_rf = []

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
        value=13,
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
        "time_bins",
        "count_data",
        "bin_width",
        "record_length",
        "pulse_profile",
        "flank",
        "time_window_width",
        "time_window_offset_signal",
        "time_window_offset_normalize",
        "tau",
        "y1",
        "y2",
        "nu",
        "yf1",
        "yf2",
        "vpp1",
        "vpp2",
        "sampling",
        "trig_interval",
        "trig_delay",
        "vpp_rf",
        "sampling_rf",
        "trig_interval_rf",
        "trig_delay_rf",
        "start_time",
        "end_time",
        "time_step",
        "main_dir",
        "main_dir_rf",
        "freq",
        "power",
    ]

    # Constructor ########################
    def __init__(self, pulse_generator, time_tagger, microwave, awg, awg_rf, **kwargs):
        super(Pulsed, self).__init__(**kwargs)
        self.pulse_generator = pulse_generator
        self.time_tagger = time_tagger
        self.microwave = microwave
        self.awg = awg
        self.awg_rf = awg_rf
        self.awg.sync_upload_trait(self, "upload_progress")
        self.awg_rf.sync_upload_trait(self, "upload_progress_rf")

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
        self.tau = np.arange(self.start_time, self.end_time, self.time_step)

        self.waves = []
        self.main_wave = ""

        self.compile_waveforms()

        self.awg.ftp_cwd = "/main" + self.main_dir
        self.awg.upload(self.waves)
        self.awg.managed_load(self.main_wave, cwd=self.main_dir)
        # self.awg.managed_load(self.main_wave)
        self.reload_awg = False

    def load_wfm_rf(self):
        self.tau = np.arange(self.start_time, self.end_time, self.time_step)

        self.waves_rf = []
        self.main_wave_rf = ""

        self.compile_waveforms()

        self.awg_rf.ftp_cwd = "/main" + self.main_dir_rf
        self.awg_rf.upload(self.waves_rf)
        self.awg_rf.managed_load(self.main_wave_rf, cwd=self.main_dir_rf)
        # self.awg.managed_load(self.main_wave)
        self.reload_awg_rf = False

    def prepare_awg(self):
        if self.reload_awg:
            self.load_wfm()
        self.awg.set_vpp(self.vpp1, 0b01)
        self.awg.set_vpp(self.vpp2, 0b10)
        self.awg.set_sampling(self.sampling)
        self.awg.set_mode("E")

    def prepare_awg_rf(self):
        if self.reload_awg_rf:
            self.load_wfm_rf()
        self.awg_rf.set_vpp(self.vpp_rf)
        self.awg_rf.set_sampling(self.sampling_rf)
        self.awg_rf.set_mode("E")

    def _wfm_button_fired(self):
        self.reload_awg = True
        self.prepare_awg()

    def _wfm_button_rf_fired(self):
        self.reload_awg_rf = True
        self.prepare_awg_rf()

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
        self.awg_rf.set_output(1)
        self.awg_rf.run()

    def shut_down(self):
        """Override for additional stuff to be executed at shut down."""
        self.pulse_generator.Light()
        self.microwave.setOutput(None, self.freq)
        self.awg.stop()
        self.awg.set_output(0b00)
        self.awg_rf.stop()
        self.awg_rf.set_output(0)

    def shut_down_finally(self):
        """Override for additional stuff to be executed finally."""
        self.pulse_generator.Light()
        self.microwave.setOutput(None, self.freq)
        self.awg.stop()
        self.awg.set_output(0b00)
        self.awg_rf.stop()
        self.awg_rf.set_output(0)

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
            self.prepare_awg_rf()  # TODO: if reload: block thread untill upload is complete

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
        # plot.tools.append(PanTool(plot))
        # plot.overlays.append(SimpleZoom(plot, enable_wheel=True)) #changed to true
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

    @on_trait_change("pulse_profile, count_data")
    def _update_pulse_value(self):
        self.pulse_plot_data.set_data("x", self.time_bins)
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
