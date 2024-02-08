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
from chaco.api import ArrayPlotData, Plot, Spectral, PlotLabel

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
from hardware.waveform import *

from analysis import fitting
from measurements.pulsed import find_laser_pulses


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

    # measurement data
    frequency = Array()
    counts = Array()
    counts_matrix = Array()
    run_time = Float(value=0.0, desc="Run time [s]", label="Run time [s]")

    # plotting
    line_label = Instance(PlotLabel)
    line_data = Instance(ArrayPlotData)
    matrix_data = Instance(ArrayPlotData)
    line_plot = Instance(Plot, editor=ComponentEditor())
    matrix_plot = Instance(Plot, editor=ComponentEditor())

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
            if self.periodic_focus:
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
            self.state = "run"
            self.apply_parameters()

            if self.run_time >= self.stop_time:
                self.state = "done"
                return

            # if pulsed, turn on sequence
            if self.pulsed:
                # ha.PulseGenerator().Sequence(100 * [ (['laser', 'aom'], self.laser), ([], self.wait), (['mw','mw_x'], self.t_pi) ])
                # ha.PulseGenerator().Sequence(100 * [ (['laser', 'aom'], self.laser), (['rf', 'mw_b', 'mw', 'aom'], self.randomTime), (['aom'], self.laser), ([], self.wait), (['mw_x'], self.t_pi) ])
                # ha.PulseGenerator().Sequence(100 * [ (['laser', 'aom'], self.laser), ([], self.wait), (['mw_x'], self.t_pi) ])
                ha.PulseGenerator().Sequence(
                    100
                    * [
                        (["B", "laser", "aom"], self.laser),
                        (
                            [
                                "B",
                            ],
                            self.wait,
                        ),
                        (["A", "B", "mw_x"], self.t_pi),
                    ]
                )
                # ha.PulseGenerator().Sequence(100 * [ (['aom'], self.t_pi),  (['mw_x','aom','laser'], self.laser), (['aom'], 6.0)])
            else:
                # ha.PulseGenerator().Continuous(['green','mw','mw_x'])
                ha.PulseGenerator().Continuous(["A", "green", "mw_x"])

            n = len(self.frequency)

            _power = self.power_p if self.pulsed else self.power

            ha.Microwave().setPower(_power)

            if not self.randomize:
                ha.Microwave().initSweep(self.frequency, _power * np.ones(n))
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

                    ha.Microwave().initSweep(_freq, _power * np.ones(n))

                ha.Microwave().resetListPos()

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

                """
                ha.Microwave().doSweep()
                
                timeout = 3.
                start_time = time.time()
                while not self._count_between_markers.ready():
                    time.sleep(0.1)
                    if time.time() - start_time > timeout:
                        print "count between markers timeout in ODMR"
                        break
                        
                counts = self._count_between_markers.getData(0)
                self._count_between_markers.clean()
                """

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
                        for m in self.measurement_list:
                            if hasattr(m, "measurement"):
                                if hasattr(m.measurement, "frequency"):
                                    m.measurement.frequency = fp
                                else:
                                    print(
                                        (
                                            m.measurement,
                                            'has no attribute called either "frequency" or "freq"',
                                        )
                                    )
                            elif hasattr(m, "frequency"):
                                m.frequency = fp
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

            ha.Microwave().setOutput(None, self.frequency[0])
            ha.PulseGenerator().Light()
            ha.Counter().clear()
        except:
            logging.getLogger().exception("Error in odmr.")
            self.state = "error"
        finally:
            ha.Microwave().setOutput(None, self.frequency[0])

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
        line_plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=32)
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
        matrix_plot = Plot(matrix_data, padding=8, padding_left=64, padding_bottom=32)
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

    # saving data

    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)

    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)

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
            HGroup(
                Item("submit_button", show_label=False),
                Item("remove_button", show_label=False),
                Item("resubmit_button", show_label=False),
                Item("priority", enabled_when='state != "run"'),
                Item("state", style="readonly"),
                Item("run_time", style="readonly", format_str="%.f"),
                Item("stop_time"),
            ),
            VGroup(
                HGroup(
                    Item("pulsed", enabled_when='state != "run"'),
                    Item("periodic_focus"),
                ),
                Tabbed(
                    VGroup(
                        HGroup(
                            Item(
                                "power",
                                height=0.2,
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
                        label="Matrix Plot",
                    ),
                ),
            ),
            Group(
                Item("line_plot", show_label=False, resizable=True),
            ),
        ),
        menubar=MenuBar(
            Menu(
                Action(action="saveLinePlot", name="SaveLinePlot (.png)"),
                Action(action="saveMatrixPlot", name="SaveMatrixPlot (.png)"),
                Action(action="save", name="Save (.pyd or .pys)"),
                Action(action="saveAll", name="Save All (.png+.pys)"),
                Action(action="export", name="Export as Ascii (.asc)"),
                Action(action="load", name="Load"),
                Action(action="_on_close", name="Quit"),
                name="File",
            )
        ),
        title="ODMR",
        width=1000,
        height=800,
        buttons=[],
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
        "__doc__",
    ]


class ODMR_AWG(ManagedJob, GetSetItemsMixin):
    """Provides ODMR measurements."""

    # starting and stopping
    keep_data = Bool(False)  # helper variable to decide whether to keep existing data
    resubmit_button = Button(
        label="resubmit",
        desc="Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.",
    )

    # measurement parameters
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
        low=1,
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

    # AWG
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
        value=20.0,
        label="Trig Interval",
        desc="Set the trigger interval for Atrigger and Btrigger of the AWG.",
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
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    waves = []

    amp1 = Range(
        low=0.0,
        high=1.0,
        value=1.0,
        desc="Waveform amplitude factor",
        label="wfm amp",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%.2f"
        ),
    )
    frequency_awg = Range(
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
            auto_set=False, enter_set=True, evaluate=float, format_str="%.2f"
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

    # measurement data
    frequency = Array()
    counts = Array()
    counts_matrix = Array()
    run_time = Float(value=0.0, desc="Run time [s]", label="Run time [s]")

    # plotting
    line_label = Instance(PlotLabel)
    line_data = Instance(ArrayPlotData)
    matrix_data = Instance(ArrayPlotData)
    line_plot = Instance(Plot, editor=ComponentEditor())
    matrix_plot = Instance(Plot, editor=ComponentEditor())

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

    def __init__(self, *args, **kwargs):
        super(ODMR_AWG, self).__init__()
        self.measurement_list = args
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

        self.pulse_generator = kwargs["pulse_generator"]
        self.time_tagger = kwargs["time_tagger"]
        self.microwave = kwargs["microwave"]
        self.awg = kwargs["awg"]
        self.awg.sync_upload_trait(self, "upload_progress")

    def _counts_matrix_default(self):
        return np.zeros((self.n_lines, len(self.frequency)))

    def _frequency_default(self):
        return np.arange(
            self.frequency_begin_p,
            self.frequency_end_p + self.frequency_delta_p,
            self.frequency_delta_p,
        )

    def _counts_default(self):
        return np.zeros(self.frequency.shape)

    # data acquisition

    # Sequence ############################
    def generate_sequence(self):
        t_pi = self.t_pi
        laser = self.laser
        wait = self.wait
        trig_interval = self.trig_interval
        sequence = []
        for i in range(len(self.frequency)):
            sequence.append((["awg", "mw_y"], trig_interval))
            sequence.append((["mw_y"], t_pi + trig_interval))
            sequence.append((["laser", "green"], laser))
            sequence.append(([], wait))
        return sequence

    def compile_waveforms(self):
        # awg frequency, duration in terms of sampling rate
        freq_awg = self.frequency_awg / self.sampling
        t_pi = self.t_pi * self.sampling / 1e9  # time in terms of point
        phase = self.phase_sin * np.pi

        # pulse objects
        mw = Sin(duration=t_pi, freq=1.0, amp=self.amp1, phase=phase)
        zero = Idle(1)
        # pulse sequence
        pulse_sequence = [zero, mw, zero]
        idleforminpointrequirement = Idle(256)  # a waveform contain min. 2400 points
        pulse_sequence = pulse_sequence + [idleforminpointrequirement]

        seqName = "ODMR"
        seq = Sequence(seqName)
        for i in range(self.frequency.size):
            mw.freq = freq_awg  # modify freq
            name = "ODMR_%03i" % i
            waves = IQWaveform(name, pulse_sequence, file_type=0)
            self.waves.append(waves[0])
            self.waves.append(waves[1])
            seq.append(waves, wait=True)

        # a list containing a SQEX object, it's kind of dummy but just to fit the original code
        self.waves.append(seq)
        # seq.name is different from seqName,
        # seq.name contained file extension (*.seqx)
        self.main_wave = (
            seq.name
        )  # specify the main seq, in this rabi case only one seq

    def prepare_awg(self):
        if self.reload_awg:
            self.load_wfm()
        self.awg.set_vpp(self.vpp1, 0b01)
        self.awg.set_vpp(self.vpp2, 0b10)
        self.awg.set_sampling(self.sampling)
        self.awg.set_mode("E")

    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        if self.periodic_focus:
            fp = self.fit_frequencies[int(self.focus_index - 1)]
            if not np.isnan(fp):
                self.frequency_begin_p = fp - self.frequency_size
                self.frequency_end_p = fp + self.frequency_size
                self.frequency_delta_p = self.frequency_step
        frequency = np.arange(
            self.frequency_begin_p,
            self.frequency_end_p + self.frequency_delta_p,
            self.frequency_delta_p,
        )

        if not self.keep_data or np.any(frequency != self.frequency):
            self.frequency = frequency
            self.counts = np.zeros(frequency.shape)
            self.run_time = 0.0

        sequence = self.generate_sequence()
        depth = find_laser_pulses(sequence)

        self.sequence = sequence
        self.depth = depth
        self.keep_data = True  # when job manager stops and starts the job, data should be kept. Only new submission should clear data.

    def _run(self):

        try:
            self.state = "run"
            self.apply_parameters()

            if self.run_time >= self.stop_time:
                self.state = "done"
                return

            ha.PulseGenerator().Sequence(self.sequence)

            n = len(self.frequency)

            _power = self.power_p

            ha.Microwave().setPower(_power)
            self.awg.set_output(0b11)
            self.awg.run()

            if not self.randomize:
                ha.Microwave().initSweep(self.frequency, _power * np.ones(n))
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

                    ha.Microwave().initSweep(_freq, _power * np.ones(n))

                ha.Microwave().resetListPos()

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

                """
                ha.Microwave().doSweep()
                
                timeout = 3.
                start_time = time.time()
                while not self._count_between_markers.ready():
                    time.sleep(0.1)
                    if time.time() - start_time > timeout:
                        print "count between markers timeout in ODMR"
                        break
                        
                counts = self._count_between_markers.getData(0)
                self._count_between_markers.clean()
                """

            if self.run_time < self.stop_time:
                self.state = "idle"
            else:
                self.state = "done"
                if self.periodic_focus:
                    fp = self.fit_frequencies[int(self.focus_index - 1)]
                    if not np.isnan(fp):
                        for m in self.measurement_list:
                            if hasattr(m, "frequency"):
                                m.frequency = self.fit_frequencies[
                                    int(self.focus_index - 1)
                                ]
                            elif hasattr(m, "freq"):
                                print("awg")
                                m.freq = self.fit_frequencies[int(self.focus_index - 1)]
                                m.freq -= m.freq_awg
                            else:
                                print(
                                    (
                                        m,
                                        'has no attribute called either "frequency" or "freq"',
                                    )
                                )

            self.awg.stop()
            self.awg.set_output(0b00)
            ha.Microwave().setOutput(None, self.frequency[0])
            ha.PulseGenerator().Light()
            ha.Counter().clear()
        except:
            logging.getLogger().exception("Error in odmr.")
            self.state = "error"
        finally:
            ha.Microwave().setOutput(None, self.frequency[0])

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
        line_plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=32)
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
        matrix_plot = Plot(matrix_data, padding=8, padding_left=64, padding_bottom=32)
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

    # saving data

    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)

    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)

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

    def _wfm_button_fired(self):
        self.reload_awg = True
        self.prepare_awg()

    def load_wfm(self):
        self.waves = []
        self.main_wave = ""
        self.compile_waveforms()
        self.awg.upload(self.waves)
        self.awg.managed_load(self.main_wave)
        self.reload_awg = False

    traits_view = View(
        VGroup(
            HGroup(
                Item("submit_button", show_label=False),
                Item("remove_button", show_label=False),
                Item("resubmit_button", show_label=False),
                Item("priority", enabled_when='state != "run"'),
                Item("state", style="readonly"),
                Item("run_time", style="readonly", format_str="%.f"),
                Item("stop_time"),
            ),
            VGroup(
                Item("periodic_focus"),
                Tabbed(
                    VGroup(
                        HGroup(
                            Item("power_p", enabled_when='state != "run"'),
                            Item("t_pi", enabled_when='state != "run"'),
                        ),
                        HGroup(
                            Item("frequency_begin_p", enabled_when='state != "run"'),
                            Item("frequency_end_p", enabled_when='state != "run"'),
                            Item("frequency_delta_p", enabled_when='state != "run"'),
                        ),
                        HGroup(
                            Item("amp1", enabled_when='state != "run"'),
                            Item("frequency_awg", enabled_when='state != "run" and'),
                            Item("phase_sin", enabled_when='state != "run" and'),
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
                        HGroup(
                            Item("reload_awg"),
                            Item("wfm_button", show_label=False),
                            Item("upload_progress", style="readonly", format_str="%i"),
                            label="AWG",
                            show_border=True,
                        ),
                        HGroup(
                            Item("vpp1", width=-50),
                            Item("vpp2", width=-50),
                            Item("sampling", width=-80),
                            Item("trig_interval", width=-80),
                            label="AWG",
                            show_border=True,
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
                        label="Matrix Plot",
                    ),
                ),
            ),
            Group(
                Item("line_plot", show_label=False, resizable=True),
            ),
        ),
        menubar=MenuBar(
            Menu(
                Action(action="saveLinePlot", name="SaveLinePlot (.png)"),
                Action(action="saveMatrixPlot", name="SaveMatrixPlot (.png)"),
                Action(action="save", name="Save (.pyd or .pys)"),
                Action(action="saveAll", name="Save All (.png+.pys)"),
                Action(action="export", name="Export as Ascii (.asc)"),
                Action(action="load", name="Load"),
                Action(action="_on_close", name="Quit"),
                name="File",
            )
        ),
        title="ODMR (AWG)",
        width=1000,
        height=800,
        buttons=[],
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
        "frequency_awg",
        "phase_sin",
        "amp1",
        "power_p",
        "frequency_begin_p",
        "frequency_end_p",
        "frequency_delta_p",
        "laser",
        "wait",
        "t_pi",
        "vpp1",
        "vpp2",
        "sampling",
        "trig_interval",
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
