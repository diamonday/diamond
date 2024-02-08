from tokenize import String

import numpy as np

fft = np.fft.fft
ifft = np.fft.ifft


from enable.api import ComponentEditor
from traits.api import (
    Bool,
    Float,
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
from hardware.waveform import Idle, IQWaveform, Sequence, Sin
from hardware.waveform610 import Waveform610
from measurements.pulsed_2awg import Pulsed
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


class NuclearRabi_e0_rfmwrf(Pulsed):
    """NuclearRabi measurement."""

    """awg520"""
    main_dir = String(
        "\nRabi", desc=".WFM/.PAT folder in AWG 610", label="Waveform Dir."
    )
    freq_awg = Range(
        low=1,
        high=1.0e9,
        value=50e6,
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
        value=1.4e9,
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
        value=0.04,
        desc="Waveform amplitude factor",
        label="wfm amp",
        auto_set=False,
        enter_set=True,
    )
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

    """awg610"""
    main_dir_rf = String(
        "\nRabi", desc=".WFM/.PAT folder in AWG 610", label="Waveform_rf Dir."
    )
    freq_awg_rf = Range(
        low=1,
        high=2.6e9,
        value=2.940e6,
        desc="AWG_rf frequency",
        label="AWG_rf frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
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
    t_pi_rf = Range(
        low=1.0,
        high=1.0e9,
        value=200e3,
        desc="length of rf pi pulse [ns]",
        label="t pi rf [ns]",
        mode="text",
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
        "freq_awg",
        "freq_awg_rf",
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
        "mw_pi",
        "t_pi_rf",
    ]

    def __init__(self, pulse_generator, time_tagger, microwave, awg, awg_rf, **kwargs):
        self.dual = False
        super(NuclearRabi_e0_rfmwrf, self).__init__(
            pulse_generator, time_tagger, microwave, awg, awg_rf, **kwargs
        )
        self.show_fit = True
        self.freq_tot = self.freq + self.freq_awg

    # Sequence ############################
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        trig_interval = self.trig_interval
        trig_delay = self.trig_delay
        trig_interval_rf = self.trig_interval_rf
        trig_delay_rf = self.trig_delay_rf

        mw_pi = self.mw_pi
        t_pi_rf = self.t_pi_rf

        sequence = [
            (["aom"], laser),
            (
                ["aom"],
                576 * 1e9 / self.sampling_rf + 450,
            ),  # to be the same with other pulse
            ([], wait),
        ]

        for t in tau:
            # if t == tau[0]:
            #    sequence.append((['ch2','awg_rf'], trig_interval_rf))
            # else:
            #    sequence.append((['awg_rf'], trig_interval_rf))
            sequence.append((["awg_rf"], trig_interval_rf))
            sequence.append(([], trig_delay_rf + t))
            sequence.append((["awg"], trig_interval))
            sequence.append(([], trig_delay + mw_pi))
            sequence.append(([], t_pi_rf + 1000))
            sequence.append((["aom", "laser"], laser))
            sequence.append(
                (["aom"], 576 * 1e9 / self.sampling_rf + 450)
            )  # for trigger dead time of awg610 and excite nulcear
            sequence.append(([], wait))
        sequence.append((["sequence"], 100))
        """
        for t in tau:
            if t == tau[0]:
                sequence.append((['rf','awg'], trig_interval))
            else:
                sequence.append((['awg'], trig_interval))
            sequence.append(([], trig_delay + mw_pi))
            sequence.append((['awg_rf'], trig_interval_rf))
            sequence.append(([], trig_delay_rf + t))
            sequence.append(([ ], 576 * 1e9 /self.sampling_rf + 450 )) #for trigger dead time of awg610
            sequence.append((['awg'], trig_interval))
            sequence.append(([], trig_delay + mw_pi))
            sequence.append((['aom', 'laser'], laser))
            sequence.append(([ ], wait))
        sequence.append((['sequence'], 100))
        """
        """
        for t in tau:
            if t == tau[0]:
                sequence.append((['rf','awg_rf'], trig_interval_rf))
                sequence.append(([], trig_delay_rf + t))
            else:
                sequence.append((['awg_rf'], trig_interval_rf+500))
                sequence.append(([], trig_delay_rf + t))
            sequence.append(([ ], 24000))
        sequence.append((['sequence'], 100))
        """
        return sequence

    def compile_waveforms(self):
        # awg frequency in terms of sampling rate
        # f1 = (self.freq1 - self.freq)/self.sampling
        f1 = self.freq_awg / self.sampling
        phase = self.phase_sin * np.pi
        mw_pi = self.mw_pi
        t_pi_rf = self.t_pi_rf

        f1_rf = self.freq_awg_rf / self.sampling_rf
        phase_rf = self.phase_sin_rf * np.pi

        # pulse objects
        mw = Sin(mw_pi, freq=f1, amp=self.amp1, phase=phase)
        zero = Idle(1)
        mw_rf = Sin(0, freq=f1_rf, amp=self.amp1_rf, phase=phase_rf)
        pi_rf = Sin(t_pi_rf, freq=f1_rf, amp=self.amp1_rf, phase=phase_rf)
        zero_rf = Idle(1)

        # pulse sequence
        pulse_sequence = [zero, mw, zero]
        carry = Idle(self.trig_interval + self.trig_delay + mw_pi)
        pulse_sequence_rf = [zero_rf, mw_rf, carry, pi_rf, zero_rf]

        seq = Sequence("NRABI")
        seq_rf = Sequence("NRABI")

        # increment microwave duration
        name = "NRABI_pi"
        waves = IQWaveform(
            name, pulse_sequence, offset=0, file_type=0, sampling=self.sampling
        )  # 0 for wfm file and 1 for pat file
        # wave1 = Waveform(name1, pulse_sequence, offset=0, file_type=0, sampling=self.sampling)# 0 for wfm file and 1 for pat file
        self.waves.append(waves[0])
        self.waves.append(waves[1])

        for i, t in enumerate(self.tau):

            t = t * self.sampling_rf / 1e9
            mw_rf.duration = t
            name_rf = "NRABI_%03i" % i
            # waves = IQWaveform(name, pulse_sequence, file_type=0) # 0 for wfm file and 1 for pat file
            wave_rf = Waveform610(
                name_rf,
                pulse_sequence_rf,
                offset=0,
                file_type=0,
                sampling=self.sampling_rf,
            )  # 0 for wfm file and 1 for pat file
            self.waves_rf.append(wave_rf)
            seq_rf.append(wave_rf, wait=True)

            seq.append(waves, wait=True)
        self.waves.append(seq)
        self.main_wave = seq.name
        self.waves_rf.append(seq_rf)
        self.main_wave_rf = seq_rf.name

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
                fit_result = fitting.fit_rabi_phase(x, y, np.sqrt(self.y1))
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
                            Item("reload_awg_rf"),
                            Item("wfm_button", show_label=False),
                            Item("wfm_button_rf", show_label=False),
                            Item("upload_progress", style="readonly", format_str="%i"),
                            Item(
                                "upload_progress_rf", style="readonly", format_str="%i"
                            ),
                            Item("main_dir", style="readonly"),
                            Item("main_dir_rf", style="readonly"),
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
                            "freq_awg_rf",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
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
                            "t_pi_rf",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%.2f" % x,
                            ),
                            width=-80,
                        ),
                        Item(
                            "start_time",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        Item(
                            "end_time",
                            editor=TextEditor(
                                auto_set=False,
                                enter_set=True,
                                evaluate=float,
                                format_func=lambda x: "%e" % x,
                            ),
                            width=-80,
                        ),
                        Item(
                            "time_step",
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
                    ),
                    HGroup(
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
        title="NuclearRabi e=0 rfmwrf(AWG610)",
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
