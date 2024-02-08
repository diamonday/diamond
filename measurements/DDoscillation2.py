import numpy as np

from traits.api import (
    Range,
    Str,
    Float,
    Bool,
    Tuple,
    Array,
    Instance,
    Enum,
    on_trait_change,
)
from traitsui.api import (
    View,
    Item,
    Tabbed,
    HGroup,
    VGroup,
    TextEditor,
)
from traitsui.menu import Action, Menu, MenuBar
from traitsui.file_dialog import save_file

from chaco.api import (
    HPlotContainer,
    Plot,
    ArrayPlotData,
    ArrayDataSource,
    LinePlot,
)
from chaco.api import Spectral, LinearMapper
from enable.api import ComponentEditor

from hardware.api import (
    PulseGenerator,
    TimeTagger,
    Microwave,
)  # , MicrowaveD, MicrowaveE, RFSource
from hardware.api import PulseGenerator
from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin, GetSetItemsHandler
from measurements.pulsed import Pulsed as measurePulsed
from analysis.fitting import find_edge

from scipy.optimize import curve_fit


# utility functions
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


def sequence_length(sequence):
    t = 0
    for c, ti in sequence:
        t += ti
    return t


def sequence_union(s1, s2):
    """
    Return the union of two pulse sequences s1 and s2.
    """
    # make sure that s1 is the longer sequence and s2 is merged into it
    if sequence_length(s1) < sequence_length(s2):
        sp = s2
        s2 = s1
        s1 = sp
    s = []
    c1, dt1 = s1.pop(0)
    c2, dt2 = s2.pop(0)
    while True:
        if dt1 < dt2:
            s.append((set(c1) | set(c2), dt1))
            dt2 -= dt1
            try:
                c1, dt1 = s1.pop(0)
            except:
                break
        elif dt2 < dt1:
            s.append((set(c1) | set(c2), dt2))
            dt1 -= dt2
            try:
                c2, dt2 = s2.pop(0)
            except:
                c2 = []
                dt2 = np.inf
        else:
            s.append((set(c1) | set(c2), dt1))
            try:
                c1, dt1 = s1.pop(0)
            except:
                break
            try:
                c2, dt2 = s2.pop(0)
            except:
                c2 = []
                dt2 = np.inf
    return s


def sequence_remove_zeros(sequence):
    fil = [x for x in sequence if x[1] != 0.0]
    return fil


def spin_state(c, dt, T, t0=0.0, t1=-1.0):
    """
    Compute the spin state from a 2D array of count data.

    Parameters:

        c    = count data
        dt   = time step
        t0   = beginning of integration window relative to the edge
        t1   = None or beginning of integration window for normalization relative to edge
        T    = width of integration window

    Returns:

        y       = 1D array that contains the spin state
        profile = 1D array that contains the pulse profile
        edge    = position of the edge that was found from the pulse profile

    If t1<0, no normalization is performed. If t1>=0, each data point is divided by
    the value from the second integration window and multiplied with the mean of
    all normalization windows.
    """

    profile = c.sum(0)
    edge = find_edge(profile)
    I = int(round(T / float(dt)))
    i0 = edge + int(round(t0 / float(dt)))
    y = np.empty((c.shape[0],))
    for i, slot in enumerate(c):
        y[i] = slot[i0 : i0 + I].sum()
    if t1 >= 0:
        i1 = edge + int(round(t1 / float(dt)))
        y1 = np.empty((c.shape[0],))
        for i, slot in enumerate(c):
            y1[i] = slot[i1 : i1 + I].sum()
        y = y / y1 * y1.mean()
    return y, profile, edge


# fitting function
def decayCosineEstimator(x, y):
    offset = y.mean()
    amp = 2**0.5 * np.sqrt(((y - offset) ** 2).sum())
    # better to do estimation of period from
    Y = np.fft.fft(y - offset)
    N = len(Y)
    D = float(x[1] - x[0])
    i = abs(Y[1 : N / 2 + 1]).argmax() + 1
    phase = np.arctan(Y[i].imag / Y[i].real)
    T = (N * D) / i
    return amp, T, phase, offset, 0


def decayCosineFunction(x, A, T, phi, C, gamma):
    func = A * np.exp(-gamma * x) * np.cos(2 * np.pi * x / float(T) + phi) + C
    return func


class DDoscHandler(GetSetItemsHandler):
    def saveColorPlot(self, info):
        filename = save_file(title="Save Color Plot")
        if filename is "":
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_color_plot(filename)

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
            info.object.save_all_figure(filename)
            info.object.save(filename)


class DDosc(measurePulsed, ManagedJob, GetSetItemsMixin):
    """
    +======================================================================================================+
    |           DDosc: oscillation with dynamical decoupling                                               |
    |           1. CPMG                                                                                    |
    |           |I|____|*x|(_|**y|_)xN|*x|____|R|                                                          |
    |                                                                                                      |
    |            2. XY4                                                                                    |
    |           |I|____|*x|(_|**y|__|**x|__|**x|__|**y|_)xN|*x|____|R|                                     |
    |                                                                                                      |
    |            3. PulPol                                                                                 |
    |           |I|____(|*x|.|**y|.|*x||*y|.|**x|.|*y|)xN____|R|                                           |
    |                                                                                                      |
    |                                                                                                      |
    |           |I|: Initialize                                                                            |
    |           |R|: Readout                                                                               |
    |           |*x|: pi/2 pulse in x direction                                                            |
    |           |**y|: pi pulse in y direction                                                             |
    |           _: tau_DD/2 from nspin transition                                                          |
    |           .: tau_DD/4 from nspin transition                                                          |
    |           ___: waiting time                                                                          |
    |                                                                                                      |
    |                                                                                                      |
    +======================================================================================================+
    """

    # measurement parameters
    laserTime = Range(
        low=1.0,
        high=100000.0,
        value=3000.0,
        desc="laser [ns]",
        label="laser [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    wait = Range(
        low=0.0,
        high=100000.0,
        value=1000.0,
        desc="wait [ns]",
        label="wait [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    frequency = Range(
        low=1,
        high=20e9,
        value=2.8705e9,
        desc="microwave frequency",
        label="frequency [Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    power = Range(
        low=-100.0,
        high=16.0,
        value=-10,
        desc="power of microwave",
        label="power[dBm]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    t_2pi_x = Range(
        low=1.0,
        high=100000.0,
        value=400.0,
        desc="rabi period 2pi pulse length (x)",
        label="2pi x [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    t_2pi_y = Range(
        low=1.0,
        high=100000.0,
        value=400.0,
        desc="rabi period 2pi pulse length (y, 90 degree)",
        label="2pi y[ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    typeDD = Enum("CPMG", "XY4", "XY8", "PulPol", desc="type of DD", label="DD type")

    tau_DD = Range(
        low=1.5,
        high=1.0e6,
        value=2.0e3,
        desc="tau ",
        label="tau DD[ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    N_start = Range(
        low=0,
        high=1000,
        value=1,
        desc="starting number of pi pulses of transition a",
        label="N start",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    N_end = Range(
        low=1,
        high=1000,
        value=10,
        desc="ending number of pi pulses of transition a",
        label="N end",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    delta_N = Range(
        low=1,
        high=100,
        value=1,
        desc="the step in the N sequence",
        label="delta N",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    N_ref = Range(
        low=1,
        high=1000,
        value=1,
        desc="number of reference pulse",
        label="N_ref",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    N_seq = Array(value=np.array((0.0, 1, 2, 3)))
    repeat = Range(
        low=0,
        high=100,
        value=0,
        desc="repetition of the pulse sequence each run",
        label="repeat",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    # result data
    pulse = Array(value=np.array((0.0, 0.0)))
    flank = Float(value=0.0)
    spin_state = Array(value=np.array((0.0, 0.0)))
    spin_state_error = Array(value=np.array((0.0, 0.0)))
    integration_width = Range(
        low=10.0,
        high=4000.0,
        value=200.0,
        desc="time window for pulse analysis [ns]",
        label="integr. width [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    position_signal = Range(
        low=-100.0,
        high=1000.0,
        value=0.0,
        desc="position of signal window relative to edge [ns]",
        label="pos. signal [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    position_normalize = Range(
        low=0.0,
        high=10000.0,
        value=2200.0,
        desc="position of normalization window relative to edge [ns]",
        label="pos. norm. [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    refBrightDark = Array(
        value=np.array([0.0, 0.0])
    )  # reference count for 0 and +-1, array([bright,dark])
    # plotting
    show_raw_data = Bool(False, label="show raw data as matrix plot")
    matrix_plot_data = Instance(ArrayPlotData)  # raw data of spin state
    matrix_plot = Instance(Plot, editor=ComponentEditor())

    pulse_plot_data = Instance(ArrayPlotData)
    pulse_plot = Instance(Plot)

    figureContainer = Instance(
        HPlotContainer
    )  # a box contains all the colormap plot, scatter plots and colorbar

    scatterPlot_Data = Instance(ArrayPlotData)  # plot data, spinstate, fit
    scatterPlot = Instance(Plot)  # scatter plot

    brightRefLineData = Instance(ArrayPlotData)
    darkRefLineData = Instance(ArrayPlotData)

    # fitting
    perform_fit = Bool(False, label="perform fit")
    fitData = Str("None", label="fit data")  # display fit data
    para_fit = Tuple(0.0, 0.0, 0.0, 0.0, 0.0)  # store fit data
    para_dev_fit = Tuple(0.0, 0.0, 0.0, 0.0, 0.0)  # store fit data

    def __init__(self):
        super(DDosc, self).__init__()
        self.check_delta_N_valid()
        # create different plots
        self._create_matrix_plot()
        self._create_pulse_plot()
        self._create_scatterPlot()
        self._create_figureContainer()

    # ==========================================================|
    #               check parameters, set devices              |
    # ==========================================================|
    @on_trait_change("N_start, N_end, delta_N")
    def check_delta_N_valid(self):
        N_start = self.N_start
        N_end = self.N_end
        delta_N = self.delta_N
        remainder = (N_end - N_start) % delta_N
        if remainder != 0:
            self.N_end += delta_N - remainder_a
        self.N_seq = np.arange(self.N_start, self.N_end + self.delta_N, self.delta_N)

    def start_up(self):
        PulseGenerator().Night()
        Microwave().setOutput(self.power, self.frequency)

    def shut_down(self):
        PulseGenerator().Light()
        Microwave().setOutput(None, self.frequency)

    def get_DDsequence(self, N):
        # N: number of DD block (int)
        tau_DD = self.tau_DD
        t_2pi_x = self.t_2pi_x
        t_2pi_y = self.t_2pi_y
        DDtype = self.typeDD

        t_pi2_x = t_2pi_x / 4.0
        t_pi_x = t_2pi_x / 2.0
        t_pi_y = t_2pi_y / 2.0

        if DDtype == "CPMG":
            tDD = tau_DD - t_pi_y / 2.0

            if tDD < 0:
                tDD = 0
            else:
                tDD = tDD

            DDseq = [(["mw_x"], t_pi2_x)]

            if N >= 0:
                Unit_y = [([], tDD), (["mw_y"], t_pi_y), ([], tDD)]
                DDseq += Unit_y * int(N)

            DDseq += [(["mw_x"], t_pi2_x)]

        elif DDtype == "XY4":
            tDD = tau_DD - (t_pi_y + t_pi_x) / 4.0

            DDseq = [(["mw_x"], t_pi2_x)]

            if tDD < 0:
                tDD = 0

            if N >= 0:
                Unit_x = [([], tDD), (["mw_x"], t_pi_x), ([], tDD)]
                Unit_y = [([], tDD), (["mw_y"], t_pi_y), ([], tDD)]
                DDseq += (Unit_x + Unit_y + Unit_y + Unit_x) * int(N // 4)

            DDseq += [(["mw_x"], t_pi2_x)]

        elif DDtype == "XY8":
            tDD = tau_DD - (t_pi_y + t_pi_x) / 4.0

            if tDD < 0:
                tDD = 0

            DDseq = [(["mw_x"], t_pi2_x)]

            if N == 0:
                DDseq = [(["mw_x"], t_pi2_x), (["mw_x"], t_pi2_x)]
            else:
                Unit_x = [([], tDD), (["mw_x"], t_pi_x), ([], tDD)]
                Unit_y = [([], tDD), (["mw_y"], t_pi_y), ([], tDD)]
                DDseq += (Unit_x + Unit_y) * 2 * int(N // 8)
                DDseq += (Unit_y + Unit_x) * 2 * int(N // 8)

            DDseq += [(["mw_x"], t_pi2_x)]

        elif DDtype == "PulPol":
            t_pi2_x = t_2pi_x / 4.0
            t_pi2_y = t_2pi_y / 4.0
            t_pi_x = t_2pi_x / 2.0
            t_pi_y = t_2pi_y / 2.0
            tDD = tau_DD - (t_pi2_y + t_pi2_x) * 2.0 - (t_pi_x + t_pi_y) * 1.0
            if tDD < 0:
                tDD = 0
            else:
                tDD = tDD
            DDseq = [
                (["mw_x"], t_pi2_x),
                ([], tDD / 4.0),
                (["mw_y"], t_pi_y),
                ([], tDD / 4.0),
                (["mw_x"], t_pi2_x),
                (["mw_y"], t_pi2_y),
                ([], tDD / 4.0),
                (["mw_x"], t_pi_x),
                ([], tDD / 4.0),
                (["mw_y"], t_pi2_y),
            ] * int(N)
        else:
            DDseq = []
        return DDseq

    def generate_sequence(self):
        laserTime = self.laserTime
        wait = self.wait

        t_2pi_x = self.t_2pi_x

        N_ref = self.N_ref
        N_seq = self.N_seq
        repeat = self.repeat
        sequence = []

        for N in N_seq:
            DDseq = self.get_DDsequence(N)
            sequence += DDseq
            sequence += [(["laser", "aom"], laserTime), ([], wait)]
        sequence = sequence * (repeat + 1)
        sequence += N_ref * ([(["laser", "aom"], laserTime), ([], wait)])  # bright
        sequence += N_ref * (
            [(["mw_x"], t_2pi_x / 2.0)] + [(["laser", "aom"], laserTime), ([], wait)]
        )  # dark

        sequence += [(["sequence"], 100)]
        return sequence

    # ==========================================================|
    #          treat raw data and store data in objects        |
    # ==========================================================|
    @on_trait_change("count_data,integration_width,position_signal,position_normalize")
    def update_spin_state(self):
        y, profile, flank = spin_state(
            c=self.count_data,
            dt=self.bin_width,
            T=self.integration_width,
            t0=self.position_signal,
            t1=self.position_normalize,
        )

        y[y == np.inf] = 0  # turn all inf into 0
        y = np.nan_to_num(y)  # turn all NN into 0

        self.spin_state = y
        self.spin_state_error = y**0.5
        self.pulse = profile
        self.flank = self.time_bins[flank]

    # get bright/dark count and put them into refBrightDark
    @on_trait_change("spin_state")
    def update_BrightDarkRef(self):
        BDRef = self.spin_state[-self.N_ref * 2 :]
        bright = BDRef[: self.N_ref]
        dark = BDRef[self.N_ref :]
        b = sum(bright) / float(len(bright))
        d = sum(dark) / float(len(dark))
        self.refBrightDark = np.array([b, d])

    # ==========================================================|
    #                    perform fitting                       |
    # ==========================================================|
    @on_trait_change("spin_state, perform_fit")
    def update_fitData(self):
        if self.perform_fit:
            N_seq = self.N_seq
            repeat = self.repeat
            spin_state = self.spin_state[: -self.N_ref * 2]
            length = int(len(spin_state) / (repeat + 1))
            ss = np.zeros(length)
            for i in range(repeat + 1):
                ss += spin_state[i * length : (i + 1) * length]
            signal = ss
            parameters_guess = tuple(decayCosineEstimator(N_seq, signal))
            fitoutput = curve_fit(decayCosineFunction, N_seq, signal, parameters_guess)
            para = tuple(fitoutput[0])
            para_dev = tuple(np.sqrt(np.diag(fitoutput[1])))
            self.fitData = "\nperiod: {}+-{}".format(
                round(para[1], 1), round(para_dev[1], 3)
            )
            self.para_fit = para
            self.para_dev_fit = para_dev

    # ==========================================================|
    #            create all the plots and container            |
    # ==========================================================|
    def _create_matrix_plot(self):
        matrix_plot_data = ArrayPlotData(image=np.zeros((self.n_laser, self.n_bins)))
        matrix_plot = Plot(
            matrix_plot_data,
            height=300,
            resizable="hv",
            padding=8,
            padding_left=64,
            padding_bottom=32,
        )
        matrix_plot.img_plot(
            "image",
            xbounds=(self.time_bins[0], self.time_bins[-1]),
            ybounds=(0, self.n_laser),
            colormap=Spectral,
        )[0]
        matrix_plot.index_axis.title = "time [ns]"
        matrix_plot.value_axis.title = "laser pulse"
        self.matrix_plot_data = matrix_plot_data
        self.matrix_plot = matrix_plot

    def _create_pulse_plot(self):
        self.pulse_plot_data = ArrayPlotData(x=self.time_bins, y=self.pulse)
        plot = Plot(self.pulse_plot_data, padding=8, padding_left=64, padding_bottom=36)
        plot.plot(("x", "y"), style="line", color="blue", name="data")
        edge_marker = LinePlot(
            index=ArrayDataSource(np.array((0, 0))),
            value=ArrayDataSource(np.array((0, 1e9))),
            color="red",
            index_mapper=LinearMapper(range=plot.index_range),
            value_mapper=LinearMapper(range=plot.value_range),
            name="marker",
        )
        plot.add(edge_marker)
        plot.index_axis.title = "time [ns]"
        plot.value_axis.title = "intensity"
        self.pulse_plot = plot

    def _create_scatterPlot(self):
        x = self.N_seq
        y = x * 0
        b = np.array([self.refBrightDark[0]] * len(x))
        d = np.array([self.refBrightDark[1]] * len(x))
        plotdata = ArrayPlotData(N=x, Nf=x, spinstatea=y, fit=y, bright=b, dark=d)
        self.scatterPlot_Data = plotdata

        plot = Plot(
            self.scatterPlot_Data,
            orientation="h",
            width=600,
            height=300,
        )
        plot.plot(
            ("N", "spinstatea"),
            type="scatter",
            marker_size=2.0,
            marker="circle",
            color=(0.1, 0.2, 0.3, 1),
            outline_color=(0.05, 0.1, 0.15, 0.7),
        )
        plot.plot(("N", "spinstatea"), type="line", color=(0.87, 0.92, 0.85, 0.6))
        plot.plot(
            ("Nf", "fit"), type="line", line_width=2.0, color=(0.3, 0.5, 0.3, 0.9)
        )
        plot.plot(("N", "bright"), type="line", line_width=2.0, color="red")
        plot.plot(("N", "dark"), type="line", line_width=2.0, color="blue")

        plot.index_axis.title = "Num. of DD block"
        plot.value_axis.title = "spinstate"
        plot.title = ""
        self.scatterPlot = plot

    def _create_figureContainer(self):
        # a box contains all the plots
        self.figureContainer = HPlotContainer(self.scatterPlot)

    # ==========================================================|
    #      update axis of the plots when relevant changes      |
    # ==========================================================|

    @on_trait_change("time_bins, n_laser")
    def update_matrix_plot_axis(self):
        self.matrix_plot.components[0].index.set_data(
            (self.time_bins[0], self.time_bins[-1]), (0.0, float(self.n_laser))
        )

    # ==========================================================|
    #          update the plot data when count changes         |
    # ==========================================================|
    @on_trait_change("count_data, show_raw_data")
    def update_matrix_plot(self):
        s = self.count_data.shape
        limit = 10000000000
        if self.show_raw_data:
            if s[0] * s[1] < limit:
                self.matrix_plot_data.set_data("image", self.count_data)

    @on_trait_change("count_data")
    def update_pulse_plot(self):
        self.pulse_plot_data.set_data("y", self.pulse)
        self.pulse_plot_data.set_data("x", self.time_bins)
        self.pulse_plot.components[1].index.set_data(np.array((self.flank, self.flank)))

    @on_trait_change("spin_state, perform_fit")
    def update_scatterPlot(self):
        x = self.N_seq
        repeat = self.repeat
        spin_state = self.spin_state[: -self.N_ref * 2]
        length = int(len(spin_state) / (repeat + 1))
        ss = np.zeros(length)
        for i in range(repeat + 1):
            ss += spin_state[i * length : (i + 1) * length]
        y = ss
        b = np.array([self.refBrightDark[0]] * len(self.N_seq))
        d = np.array([self.refBrightDark[1]] * len(self.N_seq))

        self.scatterPlot_Data.set_data("N", x)
        self.scatterPlot_Data.set_data("spinstatea", y)
        self.scatterPlot_Data.set_data("bright", b)
        self.scatterPlot_Data.set_data("dark", d)
        if self.perform_fit:
            N_seqin = np.linspace(x[0], x[-1], 201)
            para = self.para_fit
            spinstate_fit = decayCosineFunction(N_seqin, *para)
            self.scatterPlot_Data.set_data("fit", spinstate_fit)
            self.scatterPlot_Data.set_data("Nf", N_seqin)
        else:
            self.scatterPlot_Data.set_data("fit", y)
            self.scatterPlot_Data.set_data("Nf", x)

    # ==========================================================|
    #                   save data and graphs                   |
    # ==========================================================|
    def save_color_plot(self, filename):
        self.save_figure(self.figureContainer, filename)

    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)

    def save_all_figure(self, filename):
        self.save_color_plot(filename + "_DDosc_Plot.png")
        self.save_matrix_plot(filename + "_DDosc_Matrix_Plot.png")

    get_set_items = measurePulsed.get_set_items + [
        "laserTime",
        "wait",
        "frequency",
        "power",
        "t_2pi_x",
        "t_2pi_y",
        "tau_DD",
        "typeDD",
        "N_start",
        "N_end",
        "delta_N",
        "N_ref",
        "N_seq",
        "__doc__",
        "integration_width",
        "position_signal",
        "position_normalize",
        "pulse",
        "flank",
        "spin_state",
        "spin_state_error",
        "refBrightDark",
        "perform_fit",
        "para_fit",
        "para_dev_fit",
    ]

    traits_view = View(
        VGroup(
            VGroup(
                HGroup(
                    Item("submit_button", show_label=False),
                    Item("remove_button", show_label=False),
                    Item("resubmit_button", show_label=False),
                    Item("priority"),
                    Item("state", style="readonly"),
                    Item("run_time", style="readonly", format_str="%.f"),
                    Item("stop_time"),
                ),
            ),
            HGroup(
                VGroup(
                    HGroup(
                        VGroup(
                            Item("bin_width", width=-80, enabled_when='state != "run"'),
                            Item(
                                "record_length",
                                width=-80,
                                enabled_when='state != "run"',
                            ),
                            Item(
                                "integration_width",
                                width=-80,
                                enabled_when='state != "run"',
                            ),
                            Item(
                                "position_signal",
                                width=-80,
                                enabled_when='state != "run"',
                            ),
                            Item(
                                "position_normalize",
                                width=-80,
                                enabled_when='state != "run"',
                            ),
                            Item("laserTime", width=-80, enabled_when='state != "run"'),
                            Item("wait", width=-80, enabled_when='state != "run"'),
                            Item("frequency", width=-80, enabled_when='state != "run"'),
                            Item("power", width=-80, enabled_when='state != "run"'),
                        ),
                        VGroup(
                            Item("typeDD", width=-80, enabled_when='state != "run"'),
                            Item("tau_DD", width=-80, enabled_when='state != "run"'),
                            Item("t_2pi_x", width=-80, enabled_when='state != "run"'),
                            Item("t_2pi_y", width=-80, enabled_when='state != "run"'),
                            Item("N_start", width=-80, enabled_when='state != "run"'),
                            Item("N_end", width=-80, enabled_when='state != "run"'),
                            Item("delta_N", width=-80, enabled_when='state != "run"'),
                            Item("N_ref", width=-80, enabled_when='state != "run"'),
                            Item("repeat", width=-80, enabled_when='state != "run"'),
                        ),
                    ),
                    HGroup(
                        VGroup(
                            Item("perform_fit"),
                            Item("fitData", style="readonly"),
                        ),
                        show_border=False,
                    ),
                ),
                VGroup(
                    Tabbed(
                        VGroup(  # Item('cursorPosition', style='readonly'),
                            Item(
                                "figureContainer",
                                editor=ComponentEditor(),
                                show_label=False,
                                height=520,
                                resizable=True,
                            ),
                            label="graph",
                        ),
                        VGroup(
                            Item(
                                "matrix_plot",
                                editor=ComponentEditor(),
                                show_label=False,
                                height=500,
                                resizable=True,
                            ),
                            Item("show_raw_data"),
                            label="raw data",
                        ),
                        VGroup(
                            Item(
                                "pulse_plot",
                                editor=ComponentEditor(),
                                show_label=False,
                                height=500,
                                resizable=True,
                            ),
                            label="profile",
                        ),
                    ),
                ),
            ),
        ),
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
        title="DD oscillation",
        width=1250,
        resizable=True,
        handler=DDoscHandler,
    )


if __name__ == "__main__":
    pass
