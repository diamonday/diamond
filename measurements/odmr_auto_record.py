import numpy as np

from traits.api import (
    Trait,
    Instance,
    String,
    Range,
    Float,
    Int,
    Bool,
    Array,
    Button,
)
from traitsui.api import (
    View,
    Item,
    HGroup,
    VGroup,
    Tabbed,
    TextEditor,
    Group,
)
from enable.api import ComponentEditor
from chaco.api import PlotAxis, ArrayPlotData, Plot, Spectral, PlotLabel

from chaco.scales.api import CalendarScaleSystem
from chaco.scales_tick_generator import ScalesTickGenerator

from traitsui.file_dialog import save_file

from traitsui.menu import Action, Menu, MenuBar

import time
import threading
import logging

import random

from tools.emod import ManagedJob
from tools.cron import CronDaemon, CronEvent
from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import hardware.api as ha

from analysis import fitting


class ODMRHandler(GetSetItemsHandler):

    def saveLinePlot(self, info):
        filename = save_file(title="Save Line Plot")
        if filename is "":
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_line_plot(filename)

    def saveMatrixPlot(self, info):
        filename = save_file(title="Save Matrix Plot")
        if filename is "":
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_matrix_plot(filename)

    def saveDriftPlot(self, info):
        filename = save_file(title="Save Drift Plot")
        if filename is "":
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_drift_plot(filename)

    def saveAll(self, info):
        filename = save_file(title="Save All")
        if filename is "":
            return
        else:
            info.object.save_all(filename)


class ODMR(ManagedJob, GetSetItemsMixin):
    """Provides ODMR measurements."""

    # starting and stopping
    keep_data = Bool(False)  # helper variable to decide whether to keep existing data
    resubmit_button = Button(
        label="resubmit",
        desc="Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.",
    )

    # measurement parameters
    power = Range(
        low=-100.0,
        high=0.0,
        value=-30.0,
        desc="Power [dBm]",
        label="Power [dBm]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    frequency_begin = Range(
        low=1,
        high=3.3e9,
        value=2.8e9,
        desc="Start Frequency [Hz]",
        label="Begin [Hz]",
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    frequency_end = Range(
        low=1,
        high=3.3e9,
        value=2.94e9,
        desc="Stop Frequency [Hz]",
        label="End [Hz]",
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    frequency_delta = Range(
        low=1e-3,
        high=3.3e9,
        value=1.0e6,
        desc="frequency step [Hz]",
        label="Delta [Hz]",
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    t_pi = Range(
        low=1.0,
        high=100000.0,
        value=1700.0,
        desc="length of pi pulse [ns]",
        label="pi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    laser = Range(
        low=1.0,
        high=10000.0,
        value=300.0,
        desc="laser [ns]",
        label="laser [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    wait = Range(
        low=1.0,
        high=10000.0,
        value=1000.0,
        desc="wait [ns]",
        label="wait [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    pulsed = Bool(True, label="pulsed")
    stop_time = Range(
        low=1.0,
        value=np.inf,
        desc="Time after which the experiment stops by itself [s]",
        label="Stop time [s]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    power_p = Range(
        low=-100.0,
        high=20.0,
        value=-28.0,
        desc="Power Pmode [dBm]",
        label="Power Pmode[dBm]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    frequency_begin_p = Range(
        low=1,
        high=6.4e9,
        value=1.55e9,
        desc="Start Frequency Pmode[Hz]",
        label="Begin Pmode[Hz]",
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    frequency_end_p = Range(
        low=1,
        high=6.4e9,
        value=1.555e9,
        desc="Stop Frequency Pmode[Hz]",
        label="End Pmode[Hz]",
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    frequency_delta_p = Range(
        low=1e-3,
        high=3.3e9,
        value=1.0e5,
        desc="frequency step Pmode[Hz]",
        label="Delta Pmode[Hz]",
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    seconds_per_point = Range(
        low=10e-3,
        high=1,
        value=20e-3,
        desc="Seconds per point",
        label="Seconds per point",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    n_lines = Range(
        low=1,
        high=10000,
        value=50,
        desc="Number of lines in Matrix",
        label="Matrix lines",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    priority = Range(
        low=0,
        high=10,
        value=1,
        desc="priority of the job",
        label="priority",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    focus_interval = Range(
        low=1,
        high=6000,
        value=10,
        desc="Time interval between automatic focus events",
        label="Interval [m]",
        auto_set=False,
        enter_set=True,
    )
    periodic_focus = Bool(False, label="Periodic focusing")
    focus_index = Range(
        low=0,
        high=1000,
        value=1,
        desc="The peak being focused on",
        label="focus index",
        auto_set=False,
        enter_set=True,
    )
    power_p = Range(
        low=-100.0,
        high=20.0,
        value=-20,
        desc="Power Pmode [dBm]",
        label="Power Pmode[dBm]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    frequency_size = Range(
        low=1,
        high=6.4e9,
        value=1.0e6,
        desc="Radius of the frequency list[Hz]",
        label="Frequency Size[Hz]",
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    frequency_step = Range(
        low=1e-3,
        high=3.3e9,
        value=2.0e4,
        desc="Frequency step Pmode[Hz]",
        label="Delta Pmode[Hz]",
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )

    # control data fitting
    perform_fit = Bool(False, label="perform fit")
    number_of_resonances = Trait(
        "auto",
        String("auto", auto_set=False, enter_set=True),
        Int(
            10000.0,
            desc="Number of Lorentzians used in fit",
            label="N",
            auto_set=False,
            enter_set=True,
        ),
    )
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

    # fit result
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    fit_frequencies = Array(value=np.array((np.nan,)), label="frequency [Hz]")
    fit_line_width = Array(value=np.array((np.nan,)), label="line_width [Hz]")
    fit_contrast = Array(value=np.array((np.nan,)), label="contrast [%]")

    # Drift
    frequency_origin = Float(value=-1.0, label="Initial Frequency")
    frequency_drift = Array(value=np.array([0.0]))
    drift_time = Array(value=np.array([0.0]))
    forget_drift = Button(label="Forget Drift", desc="Clear all of the record.")

    # measurement data
    frequency = Array()
    counts = Array()
    counts_matrix = Array()
    run_time = Float(value=0.0, desc="Run time [s]", label="Run time [s]")

    # plotting
    line_label = Instance(PlotLabel)
    line_data = Instance(ArrayPlotData)
    matrix_data = Instance(ArrayPlotData)
    drift_data = Instance(ArrayPlotData)

    line_plot = Instance(Plot, editor=ComponentEditor())
    matrix_plot = Instance(Plot, editor=ComponentEditor())
    drift_plot = Instance(Plot, editor=ComponentEditor())

    randomize = Bool(True)
    randomize_interval = Range(
        low=1,
        high=1000,
        value=5,
        label="Randomize Interval [Runs]",
        evaluate=int,
        auto_set=False,
        enter_set=True,
    )

    def __init__(self, *args):
        super(ODMR, self).__init__()
        self.measurement_list = args
        self.drift_time[0] = time.time()
        self._create_line_plot()
        self._create_matrix_plot()
        self.on_trait_change(self._update_line_data_index, "frequency", dispatch="ui")
        self.on_trait_change(self._update_line_data_value, "counts", dispatch="ui")
        self.on_trait_change(
            self._update_line_data_fit, "fit_parameters", dispatch="ui"
        )
        self.on_trait_change(
            self._update_matrix_data_value, "counts_matrix", dispatch="ui"
        )
        self.on_trait_change(
            self._update_matrix_data_index, "n_lines,frequency", dispatch="ui"
        )
        self.on_trait_change(
            self._update_fit,
            "counts,perform_fit,number_of_resonances,threshold",
            dispatch="ui",
        )
        self.on_trait_change(
            self._update_drift_data_value, "frequency_drift", dispatch="ui"
        )
        self.on_trait_change(self._update_drift_data_index, "drift_time", dispatch="ui")

    def _counts_matrix_default(self):
        return np.zeros((self.n_lines, len(self.frequency)))

    def _frequency_default(self):
        if self.pulsed:
            return np.arange(
                self.frequency_begin_p,
                self.frequency_end_p + self.frequency_delta_p,
                self.frequency_delta_p,
            )
        else:
            return np.arange(
                self.frequency_begin,
                self.frequency_end + self.frequency_delta,
                self.frequency_delta,
            )

    def _counts_default(self):
        return np.zeros(self.frequency.shape)

    # data acquisition
    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        if self.pulsed:
            # We must not update the frequency list if we want to keep the data
            if self.periodic_focus and not self.keep_data:
                if self.focus_index > 0:
                    fp = self.fit_frequencies[int(self.focus_index - 1)]
                elif self.focus_index == 0:
                    fp = self.fit_frequencies.mean()
                if not np.isnan(fp):
                    self.frequency_begin_p = fp - self.frequency_size
                    self.frequency_end_p = fp + self.frequency_size
                    self.frequency_delta_p = self.frequency_step
            frequency = np.arange(
                self.frequency_begin_p,
                self.frequency_end_p + self.frequency_delta_p,
                self.frequency_delta_p,
            )
        else:
            frequency = np.arange(
                self.frequency_begin,
                self.frequency_end + self.frequency_delta,
                self.frequency_delta,
            )

        if not self.keep_data or np.any(frequency != self.frequency):
            self.frequency = frequency
            self.counts = np.zeros(frequency.shape)
            self.run_time = 0.0

        self.keep_data = True  # when job manager stops and starts the job, data should be kept. Only new submission should clear data.

    def _run(self):

        try:
            self.apply_parameters()
            self.state = "run"

            if self.run_time >= self.stop_time:
                self.state = "done"
                return

            # if pulsed, turn on sequence
            if self.pulsed:

                # ha.PulseGenerator().Sequence(100 * [ (['B','laser', 'aom'], self.laser), (['B',], self.wait), (['A','B','mw_x'], self.t_pi) ]) #for mw and rf circuit with awg520
                # ha.PulseGenerator().Sequence(100 * [ (['laser', 'aom'], self.laser), ([], self.wait), (['A','mw_x'], self.t_pi) ]) #for VQE mw and rf circuit without awg520
                ha.PulseGenerator().Sequence(
                    100
                    * [
                        (["laser", "aom"], self.laser),
                        ([], self.wait),
                        (["mw_all_x"], self.t_pi),
                    ]
                )  # for VQE mw and rf circuit with 3 smiq
            else:
                # ha.PulseGenerator().Continuous(['A','green','mw_x'])
                ha.PulseGenerator().Continuous(
                    ["green", "mw_all_x"]
                )  # for VQE mw and rf circuit with 3 smiq

            n = len(self.frequency)
            _power = self.power_p if self.pulsed else self.power
            # ha.Microwave().setPower(_power)
            ha.MicrowaveC().setPower(_power)  # for 3 smiq

            if not self.randomize:
                # ha.Microwave().initSweep(self.frequency, _power * np.ones(n))
                ha.MicrowaveC().initSweep(
                    self.frequency, _power * np.ones(n)
                )  # for 3 smiq
            else:
                _rand_run_remains = 0

            ha.Counter().configure(n, self.seconds_per_point, DutyCycle=0.8)

            time.sleep(0.5)

            while self.run_time < self.stop_time:
                start_time = time.time()
                if threading.currentThread().stop_request.isSet():
                    break
                if self.randomize and _rand_run_remains <= 0:
                    _rand_run_remains = self.randomize_interval
                    _rand_mapping = list(range(0, n))
                    random.shuffle(_rand_mapping)
                    _freq = [self.frequency[_ind] for _ind in _rand_mapping]
                    # ha.Microwave().initSweep(_freq, _power * np.ones(n))
                    ha.MicrowaveC().initSweep(_freq, _power * np.ones(n))  # for 3 smiq
                # ha.Microwave().resetListPos()
                ha.MicrowaveC().resetListPos()  # for 3 smiq
                _data = ha.Counter().run()
                if self.randomize:
                    _counts = np.zeros(n)
                    for _old_ind, _new_ind in enumerate(_rand_mapping):
                        logging.getLogger().debug(
                            "Data Insertion Position: " + str(_new_ind)
                        )
                        _counts[_new_ind] = _data[_old_ind]
                    _rand_run_remains -= 1
                else:
                    _counts = _data

                self.run_time += time.time() - start_time
                self.counts += _counts
                self.counts_matrix = np.vstack((_counts, self.counts_matrix[:-1, :]))
                self.trait_property_changed("counts", self.counts)

            if self.run_time < self.stop_time:
                self.state = "idle"
            else:
                self.state = "done"
                if self.periodic_focus:
                    if self.focus_index > 0:
                        fp = self.fit_frequencies[int(self.focus_index - 1)]
                    elif self.focus_index == 0:
                        fp = self.fit_frequencies.mean()
                    if not np.isnan(fp):
                        if self.frequency_origin <= 0:
                            self.frequency_origin = fp
                        else:
                            current_drift = fp - self.frequency_origin
                            self.frequency_drift = np.append(
                                self.frequency_drift, current_drift
                            )
                            self.drift_time = np.append(self.drift_time, time.time())

                        for m in self.measurement_list:
                            if hasattr(m, "frequency"):
                                m.frequency = fp
                            elif hasattr(m, "measurement"):
                                if hasattr(m.measurement, "frequency"):
                                    m.measurement.frequency = fp
                                elif hasattr(m.measurement, "freq"):
                                    m.measurement.freq = fp
                                else:
                                    print(
                                        (
                                            m.measurement,
                                            'has no attribute called either "frequency" or "freq"',
                                        )
                                    )
                            elif hasattr(m, "freq"):
                                m.freq = fp
                                m.freq -= m.freq_awg
                            else:
                                print(
                                    (
                                        m,
                                        'has no attribute called either "frequency" or "freq"',
                                    )
                                )

            # ha.Microwave().setOutput(None, self.frequency[0])
            ha.MicrowaveC().setOutput(None, self.frequency[0])  # for 3 smiq
            ha.PulseGenerator().Light()
            ha.Counter().clear()
        except:
            logging.getLogger().exception("Error in odmr.")
            self.state = "error"
        finally:
            # ha.Microwave().setOutput(None, self.frequency[0])
            ha.MicrowaveC().setOutput(None, self.frequency[0])  # for 3 smiq

    def _periodic_focus_changed(self, new):
        if not new and hasattr(self, "cron_event"):
            CronDaemon().remove(self.cron_event)
        if new:
            self.cron_event = CronEvent(
                self.submit, min=list(range(0, 60, self.focus_interval))
            )
            CronDaemon().register(self.cron_event)

    # fitting
    def _update_fit(self):
        if self.perform_fit:
            N = self.number_of_resonances
            if N != "auto":
                N = int(N)
            try:
                p = fitting.fit_multiple_lorentzians(
                    self.frequency, self.counts, N, threshold=self.threshold * 0.01
                )
            except Exception:
                logging.getLogger().debug("ODMR fit failed.", exc_info=True)
                p = np.nan * np.empty(4)
        else:
            p = np.nan * np.empty(4)
        self.fit_parameters = p
        self.fit_frequencies = p[1::3]
        self.fit_line_width = p[2::3]
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
        self.fit_contrast = contrast

    # plotting
    def _create_line_plot(self):
        line_data = ArrayPlotData(
            frequency=np.array((0.0, 1.0)),
            counts=np.array((0.0, 0.0)),
            fit=np.array((0.0, 0.0)),
        )
        line_plot = Plot(
            line_data,
            width=100,
            height=100,
            padding=8,
            padding_left=64,
            padding_right=16,
            padding_bottom=32,
        )
        line_plot.plot(("frequency", "counts"), style="line", color="blue")
        line_plot.index_axis.title = "Frequency [MHz]"
        line_plot.value_axis.title = "Fluorescence counts"
        line_label = PlotLabel(
            text="", hjustify="left", vjustify="bottom", position=[64, 128]
        )
        line_plot.overlays.append(line_label)
        self.line_label = line_label
        self.line_data = line_data
        self.line_plot = line_plot

    def _create_matrix_plot(self):
        matrix_data = ArrayPlotData(image=np.zeros((2, 2)))
        matrix_plot = Plot(
            matrix_data,
            width=100,
            height=20,
            padding=8,
            padding_left=64,
            padding_right=16,
            padding_bottom=32,
        )
        matrix_plot.index_axis.title = "Frequency [MHz]"
        matrix_plot.value_axis.title = "line #"
        matrix_plot.img_plot(
            "image",
            xbounds=(self.frequency[0], self.frequency[-1]),
            ybounds=(0, self.n_lines),
            colormap=Spectral,
        )
        self.matrix_data = matrix_data
        self.matrix_plot = matrix_plot

    def _drift_plot_default(self):
        # drift_data = ArrayPlotData(t=np.array((0., 1.)), record=np.array((0., 0.)))
        self.drift_data = self._drift_data_default()
        plot = Plot(
            self.drift_data,
            width=100,
            height=20,
            padding=8,
            padding_left=64,
            padding_right=16,
            padding_bottom=32,
        )
        plot.plot(("t", "record"), style="line", color="blue")
        bottom_axis = PlotAxis(
            plot,
            orientation="bottom",
            tick_generator=ScalesTickGenerator(scale=CalendarScaleSystem()),
        )
        plot.index_axis = bottom_axis
        plot.index_axis.title = "time"
        plot.value_axis.title = "drift [kHz]"
        # plot.legend.visible=True
        # plot.tools.append(SaveTool(plot))
        return plot

    def _drift_data_default(self):
        return ArrayPlotData(t=self.drift_time, record=self.frequency_drift)

    def _forget_drift_fired(self):
        self.frequency_origin = -1.0
        self.drift_time = np.array([time.time()])
        self.frequency_drift = np.array([0.0])

    def _perform_fit_changed(self, new):
        plot = self.line_plot
        if new:
            plot.plot(("frequency", "fit"), style="line", color="red", name="fit")
            self.line_label.visible = True
        else:
            plot.delplot("fit")
            self.line_label.visible = False
        plot.request_redraw()

    def _update_line_data_index(self):
        self.line_data.set_data("frequency", self.frequency * 1e-6)
        self.counts_matrix = self._counts_matrix_default()

    def _update_line_data_value(self):
        self.line_data.set_data("counts", self.counts)

    def _update_line_data_fit(self):
        if not np.isnan(self.fit_parameters[0]):
            self.line_data.set_data(
                "fit", fitting.NLorentzians(*self.fit_parameters)(self.frequency)
            )
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
            s = ""
            for i, fi in enumerate(f):
                s += "f %i: %.6e Hz, HWHM %.3e Hz, contrast %.1f%%\n" % (
                    i + 1,
                    fi,
                    w[i],
                    contrast[i],
                )
            self.line_label.text = s

    def _update_matrix_data_value(self):
        self.matrix_data.set_data("image", self.counts_matrix)

    def _update_matrix_data_index(self):
        if self.n_lines > self.counts_matrix.shape[0]:
            self.counts_matrix = np.vstack(
                (
                    self.counts_matrix,
                    np.zeros(
                        (
                            self.n_lines - self.counts_matrix.shape[0],
                            self.counts_matrix.shape[1],
                        )
                    ),
                )
            )
        else:
            self.counts_matrix = self.counts_matrix[: self.n_lines]
        self.matrix_plot.components[0].index.set_data(
            (self.frequency[0] * 1e-6, self.frequency[-1] * 1e-6),
            (0.0, float(self.n_lines)),
        )

    def _update_drift_data_value(self):
        if len(self.frequency_drift) == 1:
            self.drift_data.set_data("record", np.array(()))
        else:
            self.drift_data.set_data("record", self.frequency_drift * 1e-3)

    def _update_drift_data_index(self):
        if len(self.drift_time) == 0:
            self.drift_data.set_data("t", np.array(()))
        else:
            self.drift_data.set_data("t", self.drift_time)

    # saving data
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)

    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)

    def save_drift_plot(self, filename):
        self.save_figure(self.drift_plot, filename)

    def save_all(self, filename):
        self.save_line_plot(filename + "_lineplot.png")
        self.save_matrix_plot(filename + "_matrixplot.png")
        self.save(filename + ".pys")

    # react to GUI events
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

    traits_view = View(
        VGroup(
            VGroup(
                HGroup(
                    Item("submit_button", show_label=False),
                    Item("remove_button", show_label=False),
                    Item("resubmit_button", show_label=False),
                    Item("priority", width=-40, enabled_when='state != "run"'),
                    Item("state", style="readonly"),
                ),
                HGroup(
                    Item("run_time", style="readonly", format_str="%.f"),
                    Item("stop_time", width=-40),
                    Item("pulsed", enabled_when='state != "run"'),
                ),
            ),
            VGroup(
                HGroup(
                    Item("forget_drift", show_label=False),
                    Item("periodic_focus"),
                    Item(
                        "frequency_origin",
                        style="readonly",
                        visible_when="frequency_origin > 0",
                        editor=TextEditor(
                            auto_set=False,
                            enter_set=True,
                            evaluate=float,
                            format_func=lambda x: " %.8f GHz" % (x * 1e-9),
                        ),
                    ),
                ),
                Tabbed(
                    Group(
                        Item("drift_plot", show_label=False, resizable=True),
                        label="Record",
                    ),
                    VGroup(
                        HGroup(
                            Item(
                                "power",
                                width=-40,
                                enabled_when='state != "run" and pulsed == False',
                            )
                        ),
                        HGroup(
                            Item(
                                "frequency_begin",
                                enabled_when='state != "run" and pulsed == False',
                            ),
                            Item(
                                "frequency_end",
                                enabled_when='state != "run" and pulsed == False',
                            ),
                            Item(
                                "frequency_delta",
                                enabled_when='state != "run" and pulsed == False',
                            ),
                        ),
                        label="Continuous",
                    ),
                    VGroup(
                        HGroup(
                            Item(
                                "power_p",
                                enabled_when='state != "run" and pulsed == True',
                            ),
                            Item(
                                "t_pi", enabled_when='state != "run" and pulsed == True'
                            ),
                        ),
                        HGroup(
                            Item(
                                "frequency_begin_p",
                                enabled_when='state != "run" and pulsed == True',
                            ),
                            Item(
                                "frequency_end_p",
                                enabled_when='state != "run" and pulsed == True',
                            ),
                            Item(
                                "frequency_delta_p",
                                enabled_when='state != "run" and pulsed == True',
                            ),
                        ),
                        label="Pulsed",
                    ),
                    VGroup(
                        HGroup(
                            Item("frequency_size", enabled_when="not periodic_focus"),
                            Item("frequency_step", enabled_when="not periodic_focus"),
                        ),
                        HGroup(
                            Item("focus_index", enabled_when="not periodic_focus"),
                            Item("focus_interval", enabled_when="not periodic_focus"),
                        ),
                        label="Tracking",
                    ),
                    VGroup(
                        HGroup(
                            Item("randomize", enabled_when='state != "run"'),
                            Item(
                                "randomize_interval",
                                enabled_when='state != "run" and randomize == True',
                            ),
                        ),
                        HGroup(
                            Item(
                                "seconds_per_point",
                                width=-40,
                                enabled_when='state != "run"',
                            ),
                            Item("laser", width=-50, enabled_when='state != "run"'),
                            Item("wait", width=-50, enabled_when='state != "run"'),
                        ),
                        label="General",
                    ),
                    VGroup(
                        HGroup(
                            Item("perform_fit"),
                        ),
                        HGroup(
                            Item("number_of_resonances", width=-60),
                            Item("threshold", width=-60),
                            Item("n_lines", width=-60),
                        ),
                        VGroup(
                            Item("fit_contrast", style="readonly"),
                            Item("fit_line_width", style="readonly"),
                            Item("fit_frequencies", style="readonly"),
                        ),
                        label="Fitting",
                    ),
                    Group(
                        Item("matrix_plot", show_label=False, resizable=True),
                        label="Matrix plot",
                    ),
                ),
            ),
            VGroup(
                Item("line_plot", show_label=False, resizable=True),
                label="Plot",
                show_border=True,
            ),
        ),
        menubar=MenuBar(
            Menu(
                Action(action="saveLinePlot", name="Save Line Plot (.png)"),
                Action(action="saveMatrixPlot", name="Save Matrix Plot (.png)"),
                Action(action="saveDriftPlot", name="Save Drift Plot (.png)"),
                Action(action="save", name="Save (.pyd or .pys)"),
                Action(action="saveAll", name="Save All (.png+.pys)"),
                Action(action="export", name="Export as Ascii (.asc)"),
                Action(action="load", name="Load"),
                Action(action="_on_close", name="Quit"),
                name="File",
            )
        ),
        title="ODMR",
        width=500,
        height=700,
        buttons=[],
        x=0,
        y=1080 - 740,
        resizable=True,
        handler=ODMRHandler,
    )

    get_set_items = [
        "frequency",
        "counts",
        "counts_matrix",
        "fit_parameters",
        "fit_contrast",
        "fit_line_width",
        "fit_frequencies",
        "perform_fit",
        "run_time",
        "frequency_begin",
        "frequency_end",
        "frequency_delta",
        "power_p",
        "frequency_begin_p",
        "frequency_end_p",
        "frequency_delta_p",
        "laser",
        "wait",
        "pulsed",
        "t_pi",
        "seconds_per_point",
        "stop_time",
        "n_lines",
        "number_of_resonances",
        "threshold",
        "randomize",
        "randomize_interval",
        "focus_index",
        "frequency_size",
        "frequency_step",
        "periodic_focus",
        "drift_time",
        "frequency_drift",
        "frequency_origin",
        "__doc__",
    ]


if __name__ == "__main__":

    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info("Starting logger.")

    from tools.emod import JobManager

    JobManager().start()

    o = ODMR()
    o.edit_traits()
