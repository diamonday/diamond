from tokenize import String

import numpy as np

fft = np.fft.fft
ifft = np.fft.ifft

import logging

from chaco.api import (
    ArrayPlotData,
    Plot,
)
from enable.api import ComponentEditor
from traits.api import (
    Array,
    Bool,
    Range,
    String,
    on_trait_change,
)
from traitsui.api import (
    HGroup,
    Item,
    Tabbed,  # , EnumEditor, RangeEditor,
    TextEditor,
    VGroup,
    View,
    VSplit,
)
from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import analysis.fitting as fitting
from hardware.waveform import Idle, Sequence, Sin
from hardware.waveform610 import Waveform610
from measurements.pulsed_awg_rf import Pulsed
from tools.color import scheme
from tools.utility import GetSetItemsHandler

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


class NMR_awg(Pulsed):
    """NuclearRabi measurement."""

    """awg520"""

    mw_pi = Range(
        low=1.0,
        high=100000.0,
        value=600,
        desc="length of mw pi pulse [ns]",
        label="mw_pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    mw_pi_rf = Range(
        low=1.0,
        high=10e6,
        value=200e3,
        desc="length of mw pi pulse [ns]",
        label="mw_pi_rf [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    """awg610"""
    main_dir_rf = String(
        "\nRabi", desc=".WFM/.PAT folder in AWG 610", label="Waveform_rf Dir."
    )
    phase_sin_rf = Range(
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

    amp1_rf = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform_rf amplitude factor",
        label="wfm amp",
        auto_set=False,
        enter_set=True,
    )

    start_freq = Range(
        low=1,
        high=2.6e9,
        value=2.929e6,
        desc="AWG_rf frequency",
        label="start freq [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    end_freq = Range(
        low=1,
        high=2.6e9,
        value=2.945e6,
        desc="AWG_rf frequency",
        label="end freq [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    freq_step = Range(
        low=1,
        high=2.6e9,
        value=0.2e3,
        desc="AWG_rf frequency",
        label="freq step [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )

    freqs = Array()

    # fit parameters
    show_fit = Bool(False, label="show fit")
    threshold = Range(
        low=-99,
        high=99.0,
        value=-50.0,
        desc="Threshold for detection of resonances [%]. The sign of the threshold specifies whether the resonances are negative or positive.",
        label="threshold [%]",
        mode="slider",
        auto_set=False,
        enter_set=True,
    )
    """
    freq_fit   = Range(low=1, high=2.6e9, value=2.94e6, desc=' fit frequency', label='fit frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    line_width_fit   = Range(low=1, high=2.6e9, value=2.94e6, desc='fit linewidth', label='fit linewidth [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    contrast_fit = Array(value=np.array((np.nan,)), label='contrast [%]')
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    """
    # fit_parameters = np.ones(3) #changed from 4 to 3
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    freq_fit = Array(value=np.array((np.nan,)), label="frequency [Hz]")
    line_width_fit = Array(value=np.array((np.nan,)), label="line width [Hz]")
    contrast_fit = Array(value=np.array((np.nan,)), label="contrast [%]")

    get_set_items = Pulsed.get_set_items + [
        "mw_pi",
        "mw_pi_rf",
        "start_freq",
        "end_freq",
        "freq_step",
        "show_fit",
        "fit_parameters",
        "threshold",
        "freq_fit",
        "line_width_fit",
    ]

    def __init__(self, pulse_generator, time_tagger, microwave, awg_rf, **kwargs):
        self.dual = False
        super(NMR_awg, self).__init__(
            pulse_generator, time_tagger, microwave, awg_rf, **kwargs
        )
        self.show_fit = True
        self.freqs = np.arange(self.start_freq, self.end_freq, self.freq_step)

    # Sequence ############################
    def generate_sequence(self):
        self.tau
        laser = self.laser
        wait = self.wait
        trig_interval_rf = self.trig_interval_rf
        trig_delay_rf = self.trig_delay_rf

        mw_pi = self.mw_pi
        mw_pi_rf = self.mw_pi_rf

        self.freqs = np.arange(self.start_freq, self.end_freq, self.freq_step)
        freqs = self.freqs / self.sampling_rf

        sequence = [
            ([], trig_interval_rf + trig_delay_rf + mw_pi + mw_pi_rf + mw_pi - 100),
            (["aom"], laser),
            (["aom"], (576 + 512) * 1e9 / self.sampling_rf + 450),
            ([], wait),
        ]
        for i, f in enumerate(freqs):
            sequence.append((["awg_rf"], trig_interval_rf))
            sequence.append(([], trig_delay_rf))
            sequence.append((["A", "mw_x"], mw_pi))
            sequence.append(([], mw_pi_rf))
            sequence.append((["A", "mw_x"], mw_pi))
            sequence.append((["laser", "aom"], laser))
            sequence.append(
                (["aom"], (576 + 512) * 1e9 / self.sampling_rf + 450)
            )  # for trigger dead time of awg610(for awg520, we can ignore it if we set the sampling rate as 1e9)
            sequence.append(([], wait))

        sequence.append((["sequence"], 100))

        return sequence

    def compile_waveforms(self):
        # awg frequency in terms of sampling rate
        # f1 = (self.freq1 - self.freq)/self.sampling
        mw_pi = self.mw_pi
        mw_pi_rf = self.mw_pi_rf * self.sampling_rf / 1e9
        mw_pi_rft = mw_pi * self.sampling_rf / 1e9

        self.freqs = np.arange(self.start_freq, self.end_freq, self.freq_step)
        freqs = self.freqs / self.sampling_rf

        phase_rf = self.phase_sin_rf * np.pi

        # pulse objects
        zero_rf = Idle(1)

        seq_rf = Sequence("NMR")

        carry = Idle(0)
        carry.duration = int(mw_pi_rft)

        # increment microwave duration

        for i, f in enumerate(freqs):

            mw_rf = Sin(mw_pi_rf, freq=f, amp=self.amp1_rf, phase=phase_rf)
            name_rf = "NMR_%03i" % i
            # waves = IQWaveform(name, pulse_sequence, file_type=0) # 0 for wfm file and 1 for pat file
            pulse_sequence_rf = [carry, mw_rf, zero_rf]
            wave_rf = Waveform610(
                name_rf,
                pulse_sequence_rf,
                offset=0,
                file_type=0,
                sampling=self.sampling_rf,
            )  # 0 for wfm file and 1 for pat file
            self.waves_rf.append(wave_rf)
            seq_rf.append(wave_rf, wait=True)

        self.waves_rf.append(seq_rf)
        self.main_wave_rf = seq_rf.name

    def load_wfm_rf(self):
        self.tau = np.arange(self.start_time, self.end_time, self.time_step)
        self.freqs = np.arange(self.start_freq, self.end_freq, self.freq_step)

        self.waves_rf = []
        self.main_wave_rf = ""

        self.compile_waveforms()

        self.awg_rf.ftp_cwd = "/main" + self.main_dir_rf
        self.awg_rf.upload(self.waves_rf)
        self.awg_rf.managed_load(self.main_wave_rf, cwd=self.main_dir_rf)
        # self.awg.managed_load(self.main_wave)
        self.reload_awg_rf = False

    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        self.tau = np.arange(self.start_time, self.end_time, self.time_step)
        self.freqs = np.arange(self.start_freq, self.end_freq, self.freq_step)
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
            self.prepare_awg_rf()  # TODO: if reload: block thread untill upload is complete

        self.sequence = sequence
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.keep_data = True  # when job manager stops and starts the job, data should be kept. Only new submission should clear data.

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
        plot.index_axis.title = "frequency [Hz]"
        plot.value_axis.title = "intensity [a.u.]"
        # plot.tools.append(PanTool(plot))
        # plot.overlays.append(SimpleZoom(plot, enable_wheel=True)) #changed to true
        self.line_plot_data = line_plot_data
        self.line_plot = plot

    @on_trait_change("count_data")
    def _update_line_plot_index(self):
        self.line_plot_data.set_data("index", self.freqs)

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
                "fit", fitting.NLorentzians(*self.fit_parameters)(self.freqs)
            )

    ################## Fit ########################
    @on_trait_change("show_fit")
    def _plot_fit(self):
        plot = self.line_plot
        if self.show_fit == False:
            while len(plot.components) > 1:
                plot.remove(plot.components[-1])
        else:
            self.line_plot_data.set_data(
                "fit", fitting.NLorentzians(*self.fit_parameters)(self.freqs)
            )
            plot.plot(
                ("index", "fit"), style="line", line_width=2, color=scheme["fit 1"]
            )
        plot.request_redraw()

    @on_trait_change("y1, show_fit", "threshold")
    def _update_fit_parameters(self):
        if self.y1 is None:
            return
        else:
            self.y1.mean()

            x = self.freqs
            # y = self.y1 - y_offset
            y = self.y1

            try:
                # print(fitting.CosinusNoOffsetEstimator)
                # amp, period, phi = fitting.fit(x, y, fitting.CosinusNoOffset, fitting.CosinusNoOffsetEstimator)
                p = fitting.fit_multiple_lorentzians(
                    x, y, 1, threshold=self.threshold * 0.01
                )  # number of resonance is set as 1
                # print('fit success')
                # print(p)

            except Exception:
                logging.getLogger().debug("NMR fit failed.", exc_info=True)
                p = np.nan * np.empty(4)

            self.fit_parameters = p
            self.freq_fit = p[1::3] / 1e6  # MHz unit
            self.line_width_fit = p[2::3]
            N = 1
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
            self.contrast_fit = contrast

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
                                "frequency",
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
                            Item("reload_awg_rf"),
                            Item("wfm_button_rf", show_label=False),
                            Item(
                                "upload_progress_rf", style="readonly", format_str="%i"
                            ),
                            Item("main_dir_rf", style="readonly"),
                            label="AWG rf",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item(
                            "mw_pi",
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
                        # Item('freq_tot', style='readonly', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:' %.4f GHz' % (x*1e-9))),
                        Item(
                            "phase_sin_rf",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%.2f" % x,
                            ),
                            width=-80,
                        ),
                        Item(
                            "amp1_rf",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%.2f" % x,
                            ),
                            width=-80,
                        ),
                        Item(
                            "mw_pi_rf",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%.2f" % x,
                            ),
                            width=-80,
                        ),
                        Item(
                            "start_freq",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        Item(
                            "end_freq",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        Item(
                            "freq_step",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        label="Waveform_rf",
                        show_border=True,
                    ),
                    HGroup(
                        Item("show_fit"),
                        Item("threshold"),
                        Item(
                            "freq_fit",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.6f MHz" % x,
                            ),
                        ),
                        Item(
                            "line_width_fit",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.1f Hz" % x,
                            ),
                        ),
                        Item(
                            "contrast_fit",
                            style="readonly",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: " %.1f% % " % x,
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
                    ),
                    HGroup(
                        HGroup(
                            Item("vpp_rf", width=-50),
                            Item("sampling_rf", width=-80),
                            Item("trig_interval_rf", width=-80),
                            Item("trig_delay_rf", width=-80),
                            label="AWG_rf",
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
        title="NMR smiq awg_rf (AWG610)",
        buttons=[],
        resizable=True,
        width=1400,
        height=800,
        handler=PulsedToolHandler,
        menubar=MenuBar(
            Menu(
                Action(action="load", name="Load"),
                Action(action="save", name="Save (.pyd or .pys)"),
                Action(action="saveLinePlot", name="SaveLinePlot (.png)"),
                Action(action="saveMatrixPlot", name="SaveMatrixPlot (.png)"),
                Action(action="saveColorPlot", name="SavePlot (.png)"),
                Action(action="saveAll", name="Save All (.png+.pys)"),
                Action(action="_on_close", name="Quit"),
                name="File",
            ),
        ),
    )
