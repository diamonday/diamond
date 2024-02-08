import numpy as np
import logging
import time
import traceback

from traits.api import (
    String,
    Range,
    Str,
    Float,
    Bool,
    Array,
    Instance,
    on_trait_change,
    Button,
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


from hardware.api import (
    PulseGenerator,
    TimeTagger,
    Microwave,
)  # , MicrowaveD, MicrowaveE, RFSource
from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin, GetSetItemsHandler


from .pulsed_singleshot.common import *
from .pulsed_singleshot.ui import Histogram


class PolarizationHandler(GetSetItemsHandler):
    def save_histogram(self, info):
        filename = save_file(title="Save Histogram")
        if not filename:
            return
        else:
            if filename.find(".png") == -1:
                filename = filename + ".png"
            info.object.save_histogram(filename)

    def saveAll(self, info):
        filename = save_file(title="Save All")
        if filename is "":
            return
        else:
            # info.object.save_all_figure(filename)
            info.object.save(filename)
        info.object.save_histogram(filename + "_histogram.png")


# measurePulsed,
class Polarization_QND(ManagedJob, GetSetItemsMixin):

    time_tagger = TimeTagger
    pg = PulseGenerator()

    # Reduced time for calling pg.Sequence()
    use_stored_sequence = Bool(
        False,
        desc="Save time for generating binary sequence",
        label="Use Stored Sequence",
    )
    sequence_path = String("", desc="Sequence folder", label="Sequence Folder")
    updateSeq = Button(desc="Update current sequence", label="Update")
    BinSeq = ["", ""]

    # Readout
    frequency = Range(
        low=1,
        high=20e9,
        value=1.454545e9,
        desc="microwave frequency",
        label="frequency mw[Hz]",
        mode="text",
        auto_set=False,
        enter_set=True,
        editor=TextEditor(
            auto_set=False, enter_set=True, evaluate=float, format_str="%e"
        ),
    )
    power_mw = Range(
        low=-100.0,
        high=16.0,
        value=-10,
        desc="power of microwave",
        label="power mw[dBm]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    t_2pi_x = Range(
        low=1.0,
        high=100000.0,
        value=200.0,
        desc="rabi period 2pi pulse length (x)",
        label="2pi x [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    t_2pi_y = Range(
        low=1.0,
        high=100000.0,
        value=200.0,
        desc="rabi period 2pi pulse length (y, 90 degree)",
        label="2pi y[ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    read_time = Range(
        low=0.0001,
        high=1,
        value=0.005,
        desc="Read Time [s]",
        label="Read Time [s]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    read_runs_round = Range(
        low=1,
        high=100000,
        value=1000,
        desc="Read Runs Per Round",
        label="Runs Per Round",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    read_count_low = Range(
        low=0,
        value=200,
        high=20000,
        desc="Min Count",
        label="Min Count",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    read_count_high = Range(
        low=1,
        value=400,
        high=20000,
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

    save_time_trace = Bool(False, desc="Save Time Trace", label="Save Time Trace")
    time_trace_path = String("", desc="Time Trace Folder", label="Time Trace Folder")

    read_delay = Range(
        low=0.0,
        high=100.0,
        value=1.0,
        desc="Read Delay [s]",
        label="Read Delay [s]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    initLaser_length = Range(
        low=0.0,
        high=10000.0,
        value=300.0,
        desc="Init Laser Length",
        label="Init Laser Length [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    reInitLaser_length = Range(
        low=0.0,
        high=10000.0,
        value=300.0,
        desc="RE-init Laser Length",
        label="Re-init Laser Length [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    laser_length = Range(
        low=0.0,
        high=10000.0,
        value=300.0,
        desc="Readout Laser Length",
        label="Laser Length [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    wait_length = Range(
        low=0.0,
        high=10000.0,
        value=0.0,
        desc="wait Length",
        label="wait Length [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    tau_DD = Range(
        low=1.5,
        high=1.0e6,
        value=2.0e3,
        desc="tau DD",
        label="tau DD[ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    tau_f = Range(
        low=1.5,
        high=1.0e6,
        value=2.0e3,
        desc="tau f",
        label="tau f[ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    tau_f_ref = Range(
        low=1.5,
        high=1.0e6,
        value=2.0e3,
        desc="tau",
        label="tau f ref[ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    N_DD = Range(
        low=1,
        high=1000,
        value=1,
        desc="number of DD block",
        label="N DD",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    repetition = Range(
        low=0,
        high=100000,
        value=10,
        desc="repetition",
        label="repetition",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    pulpol = Bool(False, desc="Use PulPol for initialization", label="Use PulPol")
    tau_PP = Range(
        low=1.5,
        high=1.0e6,
        value=2.0e3,
        desc="tau PulPol",
        label="tau PulPol[ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    t_rabi = Range(
        low=0.0,
        high=100000.0,
        value=200.0,
        desc="MW period before PulPol",
        label="t rabi [ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    laser_length_PP = Range(
        low=1.0,
        high=100000.0,
        value=300.0,
        desc="laser length PulPol [ns]",
        label="laser PulPol[ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    wait_length_PP = Range(
        low=0.0,
        high=100000.0,
        value=300.0,
        desc="wait length PulPol [ns]",
        label="wait PulPol[ns]",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    repetition_PP = Range(
        low=0,
        high=100000,
        value=10,
        desc="repetition PulPol",
        label="repetition PulPol",
        mode="text",
        auto_set=False,
        enter_set=True,
    )
    N_PP = Range(
        low=1,
        high=1000,
        value=1,
        desc="number of PulPol block",
        label="N PP",
        mode="text",
        auto_set=False,
        enter_set=True,
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

    # Visualization
    histogram = Instance(Histogram, factory=Histogram, kw={"dual": True})

    visualization_view = Tabbed(
        Group(
            Item("histogram", style="custom", show_label=False),
            label="Histogram",
            show_border=True,
        ),
    )

    # Dummy Values
    runs = 0
    data_bins = Array()
    count_data = Array()
    sequence = []
    keep_data = False

    n_batch = Range(
        low=0,
        value=200,
        high=20000,
        desc="# runs per batch",
        label="# runs per batch",
        mode="text",
        auto_set=False,
        enter_set=True,
    )

    T_seq = Str("0.0 ns", label="Sequence time")
    T_seq_ref = Str("0.0 ns", label="Reference Sequence time")
    n_laser = 0
    n_laser_ref = 0

    def __init__(self):
        super(Polarization_QND, self).__init__()

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

    # ==========================================================|
    #               check parameters, set devices              |
    # ==========================================================|
    def start_up(self):
        PulseGenerator().Night()
        Microwave().setOutput(self.power_mw, self.frequency)
        # MicrowaveD().setOutput(self.power_rf, self.frequency_rf)

    def shut_down(self):

        PulseGenerator().Light()
        Microwave().setOutput(None, self.frequency)
        # MicrowaveD().setOutput(None, self.frequency_rf)

    def generate_sequence(self):

        # Laser time for initialization
        initLaser_length = self.initLaser_length
        reInitLaser_length = self.reInitLaser_length
        # Laser time for readout
        laser_length = self.laser_length
        wait_length = self.wait_length

        # MW time
        t_2pi_x = self.t_2pi_x
        t_2pi_y = self.t_2pi_y
        t_pi2_x = t_2pi_x / 4.0
        t_pi2_y = t_2pi_y / 4.0
        t_pi_x = t_2pi_x / 2.0
        t_pi_y = t_2pi_y / 2.0

        # Corr sequence
        tau_DD = self.tau_DD
        tau_f = self.tau_f
        tau_f_ref = self.tau_f_ref

        laser_length_PP = self.laser_length_PP
        wait_length_PP = self.wait_length_PP
        tau_PP = self.tau_PP
        t_rabi = self.t_rabi
        repetition_PP = self.repetition_PP
        N_PP = self.N_PP

        N_DD = self.N_DD
        repetition = self.repetition

        # Pi/2 pulses
        PI2_X = [(["mw_x"], t_pi2_x)]
        PI2_Y = [(["mw_y"], t_pi2_y)]

        # Laser initialization at the very beginning
        Init = [(["aom"], initLaser_length), ([], wait_length)]

        ############### PulPol ###############
        # DD sequence time for the SWAP gate
        tPP = tau_PP - (t_pi2_y + t_pi2_x) * 2.0 - (t_pi_x + t_pi_y) * 1.0
        if tPP < 0:
            tPP = 0

        Unit_Y_PP = [([], tPP / 4.0), (["mw_y"], t_pi_y), ([], tPP / 4.0)]
        Unit_X_PP = [([], tPP / 4.0), (["mw_x"], t_pi_x), ([], tPP / 4.0)]

        PPseq = (PI2_X + Unit_Y_PP + PI2_X + PI2_Y + Unit_X_PP + PI2_Y) * N_PP
        Init_PP = [
            (["aom"], laser_length_PP),
            ([], wait_length_PP),
        ]

        MW_PP = [(["mw_x"], t_rabi), ([], wait_length_PP)]
        PulPol0 = (PPseq + Init_PP) * repetition_PP
        PulPol1 = MW_PP + (PPseq + Init_PP) * repetition_PP

        ############### Corr Sequence ###############
        # Interrogation time
        tDD = tau_DD - (t_pi_y + t_pi_x) / 4.0
        if tDD < 0:
            tDD = 0

        # Free evolution time
        tf = tau_f - t_pi_y
        if tf < 0:
            tf = 0

        tfr = tau_f_ref - t_pi_y
        if tfr < 0:
            tfr = 0

        Unit_X = [([], tDD), (["mw_x"], t_pi_x), ([], tDD)]
        [([], tDD), (["mw_y"], t_pi_y), ([], tDD)]

        DDseq = PI2_X + (Unit_X) * int(N_DD) + PI2_Y
        Corrseq = [
            ([], tf / 2.0),
            (["mw_y"], t_pi_y),
            ([], tf / 2.0),
        ]
        Corrseq_ref = [
            ([], tfr / 2.0),
            (["mw_y"], t_pi_y),
            ([], tfr / 2.0),
        ]
        Readout = [
            (["laser", "aom"], laser_length),
        ]
        # reInit = [(['aom'], 500), ([], 300)]
        reInit = [(["aom"], reInitLaser_length), ([], wait_length)]

        # Make both sequence durations match
        delay = 0
        delay_ref = 0
        if tf > tfr:
            delay_ref = tf - tfr
        elif tfr > tf:
            delay = tfr - tf
        Delay = [([], delay)]
        Delay_ref = [([], delay_ref)]

        Start = [(["sequence"], 100)]

        # sequence0 = Init + (DDseq + Corrseq_ref + DDseq + Readout + Delay_ref)*int(repetition)
        if self.pulpol:
            # sequence0 = Init + PulPol0 + Init + (reInit + DDseq + Corrseq_ref + DDseq + Readout + Delay_ref)*int(repetition)
            Delay_PP = [([], t_rabi + wait_length_PP)]
            sequence0 = Init + (
                Delay_PP
                + PulPol0
                + Init
                + DDseq
                + Corrseq_ref
                + DDseq
                + Readout
                + reInit
                + Delay_ref
            ) * int(repetition)
            if t_rabi == "0":
                # sequence1 = Init + PulPol0 + Init + (reInit + DDseq + Corrseq + DDseq + Readout + Delay)*int(repetition)
                sequence1 = Init + (
                    PulPol0 + Init + DDseq + Corrseq + DDseq + Readout + reInit + Delay
                ) * int(repetition)
            else:
                # sequence1 = Init + PulPol1 + Init + (reInit + DDseq + Corrseq + DDseq + Readout + Delay)*int(repetition)
                sequence1 = Init + (
                    PulPol1 + Init + DDseq + Corrseq + DDseq + Readout + reInit + Delay
                ) * int(repetition)
        else:
            sequence0 = Init + (
                DDseq + Corrseq_ref + DDseq + Readout + reInit + Delay_ref
            ) * int(repetition)
            sequence1 = Init + (
                DDseq + Corrseq + DDseq + Readout + reInit + Delay
            ) * int(repetition)

        return Start + sequence0, sequence1

    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""

        data_bins = np.arange(
            self.read_count_low,
            self.read_count_high + self.read_count_delta,
            self.read_count_delta,
        )
        count_data = np.zeros((2, data_bins.size + 2))

        if self.keep_data and np.all(
            data_bins == self.data_bins
        ):  # if the count binning is the same as previous, keep existing data
            pass
        else:
            self.run_time = 0.0
            self.runs = 0

            self.data_bins = data_bins
            self.count_data = count_data

        if self.use_stored_sequence:
            """
            with open(self.sequence_path + '\BinSeq0.bin', 'rb') as bfile:
                self.BinSeq[0] = bfile.read()
            with open(self.sequence_path + '\BinSeq1.bin', 'rb') as bfile:
                self.BinSeq[1] = bfile.read()
            """
            with open(self.sequence_path + "\BinSeq.bin", "rb") as bfile:
                self.BinSeq = bfile.read()
        else:
            self.sequence = self.generate_sequence()

        self.keep_data = True

    def _run(self):
        _exit_state = ["error", "idle", "done", "error"]
        _exit_reason = 0

        n_bins = 1
        tt_ch_apd = 0
        tt_ch_laser = 2
        tt_ch_seq = 3
        try:
            self.state = "run"
            self.apply_parameters()

            self.start_up()

            # _counter = self.time_tagger.Counter(ch_apd, int(self.read_time * 1e12), self.read_runs_round)
            tagger = self.time_tagger.Pulsed(
                n_bins,
                int(np.round(self.laser_length * 1000)),
                self.n_laser + self.n_laser_ref,
                tt_ch_apd,
                tt_ch_laser,
                tt_ch_seq,
            )

            _sum = 0
            _sum_old = 0
            _sum_ref = 0
            _sum_ref_old = 0
            _wait_time = 2e-3  # 2 ms

            self.pg.Night()
            if self.use_stored_sequence:
                self.pg.halt()
                self.pg.loadPages(self.BinSeq)
                self.pg.run(triggered=False)
            else:
                self.pg.Sequence(self.sequence[0], loop=True)

            tagger.setMaxCounts(self.n_batch)

            while self.run_time < self.stop_time:

                _iter = 0

                # Wait for completion of a measurement run
                while tagger.getCounts() < self.n_batch and _iter < 50001:
                    time.sleep(_wait_time)

                    _iter += 1

                if _iter == 50001:
                    logging.getLogger().warn("Singleshot Measurement Timed out.")
                    _exit_reason = 3

                    break

                _start_time = time.time()

                # time.sleep(self.read_delay)
                # time.sleep(self.read_time * self.read_runs_round * 1.1)
                # _data = _counter.getData()
                _data = np.array(tagger.getData()).flatten()
                tagger.clear()

                # print('seq:',time.time() - _start_time)

                # Counts per run
                _sum_ref += _data[: self.n_laser_ref].sum()
                _sum += _data[self.n_laser_ref : self.n_laser + self.n_laser_ref].sum()
                ref = _sum_ref - _sum_ref_old
                data = _sum - _sum_old

                # if self.save_time_trace:
                #    _path = os.path.normpath(self.time_trace_path) + "/" + "Ref_%i.npy" % self.runs
                #
                #    np.save(_path, _data)

                if ref < self.read_count_low:
                    pass
                    # self.count_data[0][0] += 1
                elif ref > self.read_count_high:
                    self.count_data[0][-1] += 1
                else:
                    _ind = (ref - self.read_count_low) // self.read_count_delta
                    self.count_data[0][_ind + 1] += 1

                if data < self.read_count_low:
                    pass
                    # self.count_data[1][0] += 1
                elif data > self.read_count_high:
                    self.count_data[1][-1] += 1
                else:
                    _ind = (data - self.read_count_low) // self.read_count_delta
                    self.count_data[1][_ind + 1] += 1

                _sum_old = _sum
                _sum_ref_old = _sum_ref

                # print('hist:',time.time() - _start_time)

                """
                for i in range(0, self.read_runs_round):
                    #_sum = _data[i]
                    _sum = _data[i]


                    if _sum < self.read_count_low:
                        self.count_data[0][0] += 1
                    elif _sum > self.read_count_high:
                        self.count_data[0][-1] += 1
                    else:
                        _ind = (_sum - self.read_count_low) // self.read_count_delta
                        
                        self.count_data[0][_ind + 1] += 1
                
                if self.use_stored_sequence:
                    self.pg.halt()
                    self.pg.loadPages(self.BinSeq[1])
                    self.pg.run(triggered=False)
                else:
                    self.pg.Sequence(self.sequence[1], loop=True)
                
                time.sleep(self.read_delay)
                
                time.sleep(self.read_time * self.read_runs_round * 1.1)
                
                _data = _counter.getData()

                if self.save_time_trace:
                    _path = os.path.normpath(self.time_trace_path) + "/" + "Polar_%i.npy" % self.runs
                    
                    np.save(_path, _data)

                for i in range(0, self.read_runs_round):
                    _sum = _data[i]

                    if _sum < self.read_count_low:
                        self.count_data[1][0] += 1
                    elif _sum > self.read_count_high:
                        self.count_data[1][-1] += 1
                    else:
                        _ind = (_sum - self.read_count_low) // self.read_count_delta
                        
                        self.count_data[1][_ind + 1] += 1
                """

                self.runs += 1

                self.trait_property_changed("count_data", self.count_data)

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
            # del _counter
            del tagger
            self.pg.Light()

        self.state = _exit_state[_exit_reason]

    def _updateSeq_fired(self):
        self.sequence = self.generate_sequence()

        """
        BinSeq0 = self.pg.convertSequenceToBinary(self.sequence[0],loop=True)
        BinSeq1 = self.pg.convertSequenceToBinary(self.sequence[1],loop=True)

        with open(self.sequence_path + '\BinSeq0.bin', 'wb') as bfile:
            bfile.write(BinSeq0)
        with open(self.sequence_path + '\BinSeq1.bin', 'wb') as bfile:
            bfile.write(BinSeq1)
        """

        BinSeq = self.pg.convertSequenceToBinary(
            self.sequence[0] + self.sequence[1], loop=True
        )
        with open(self.sequence_path + "\BinSeq.bin", "wb") as bfile:
            bfile.write(BinSeq)

        print("Update Succeed!")

        T_seq = 0
        T_seq_ref = 0
        n_laser = 0
        n_laser_ref = 0

        for ch, t in self.sequence[0]:
            T_seq_ref += t
            if "laser" in ch:
                n_laser_ref += 1
        for ch, t in self.sequence[1]:
            T_seq += t
            if "laser" in ch:
                n_laser += 1

        self.T_seq = "%.4f ms" % (T_seq * 1e-6)
        self.T_seq_ref = "%.4f ms" % (T_seq_ref * 1e-6)
        self.n_laser = n_laser
        self.n_laser_ref = n_laser_ref

    # ==========================================================|
    #          treat raw data and store data in objects        |
    # ==========================================================|
    @on_trait_change("data_bins")
    def update_histogram(self):
        if len(self.data_bins) == 0:
            return

        self.histogram.update_data("x", self.data_bins)

        if self.runs and len(self.count_data):
            self.histogram.update_data("y1", self.count_data[0])
            self.histogram.update_data("y2", self.count_data[1])

    @on_trait_change("count_data")
    def update_count_data_change(self):
        if len(self.count_data) == 0:
            return

        self.histogram.update_data("y1", self.count_data[0])
        self.histogram.update_data("y2", self.count_data[1])

    # Plot Saving Functions
    def save_histogram(self, filename):
        self.save_figure(self.histogram.plot, filename)

    get_set_items = [
        "pulpol",
        "tau_PP",
        "t_rabi",
        "laser_length_PP",
        "repetition_PP",
        "N_PP",
        "wait_length_PP",
        "use_stored_sequence",
        "sequence_path",
        "frequency",
        "power_mw",
        "t_2pi_x",
        "t_2pi_y",
        "tau_DD",
        "tau_f",
        "tau_f_ref",
        "N_DD",
        "repetition",
        "reInitLaser_length",
        "initLaser_length",
        "laser_length",
        "wait_length",
        "read_time",
        "read_runs_round",
        "read_count_low",
        "read_count_high",
        "read_count_delta",
        "save_time_trace",
        "time_trace_path",
        "__doc__",
        "runs",
        "data_bins",
        "count_data",
        "sequence",
        "T_seq",
        "T_seq_ref",
        "keep_data",
        "n_batch",
        "n_laser",
        "n_laser_ref",
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
                    VGroup(
                        Item(
                            "initLaser_length", width=-80, enabled_when='state != "run"'
                        ),
                        Item(
                            "reInitLaser_length",
                            width=-80,
                            enabled_when='state != "run"',
                        ),
                        Item("laser_length", width=-80, enabled_when='state != "run"'),
                        Item("wait_length", width=-80, enabled_when='state != "run"'),
                        label="Laser",
                        show_border=True,
                    ),
                    VGroup(
                        Item("frequency", width=-80, enabled_when='state != "run"'),
                        Item("power_mw", width=-80, enabled_when='state != "run"'),
                        Item("t_2pi_x", width=-80, enabled_when='state != "run"'),
                        Item("t_2pi_y", width=-80, enabled_when='state != "run"'),
                        label="Microwave",
                        show_border=True,
                    ),
                    VGroup(
                        Item("tau_DD", width=-80, enabled_when='state != "run"'),
                        Item("tau_f", width=-80, enabled_when='state != "run"'),
                        Item("tau_f_ref", width=-80, enabled_when='state != "run"'),
                        Item("N_DD", width=-80, enabled_when='state != "run"'),
                        Item("repetition", width=-80, enabled_when='state != "run"'),
                        label="Corr Sequence",
                        show_border=True,
                    ),
                    VGroup(
                        Item("pulpol", width=-80, enabled_when='state != "run"'),
                        Item(
                            "laser_length_PP", width=-80, enabled_when='state != "run"'
                        ),
                        Item("wait_length", width=-80, enabled_when='state != "run"'),
                        Item("t_rabi", width=-80, enabled_when='state != "run"'),
                        Item("tau_PP", width=-80, enabled_when='state != "run"'),
                        Item("N_PP", width=-80, enabled_when='state != "run"'),
                        Item("repetition_PP", width=-80, enabled_when='state != "run"'),
                        label="PulPol Sequence",
                        show_border=True,
                    ),
                    VGroup(
                        Item("read_time", width=-80, enabled_when='state != "run"'),
                        Item(
                            "read_runs_round", width=-80, enabled_when='state != "run"'
                        ),
                        Item(
                            "read_count_low", width=-80, enabled_when='state != "run"'
                        ),
                        Item(
                            "read_count_high", width=-80, enabled_when='state != "run"'
                        ),
                        Item(
                            "read_count_delta", width=-80, enabled_when='state != "run"'
                        ),
                        Item("n_batch", width=-80, enabled_when='state != "run"'),
                        label="Photon Counter",
                        show_border=True,
                    ),
                    VGroup(
                        Item("T_seq", width=-80, style="readonly"),
                        Item("T_seq_ref", width=-80, style="readonly"),
                        show_border=True,
                    ),
                ),
                VGroup(
                    VGroup(
                        HGroup(
                            Item("use_stored_sequence", enabled_when='state != "run"'),
                            Item(
                                "sequence_path",
                                width=0.1,
                                enabled_when='state != "run"',
                            ),
                            Item("updateSeq", enabled_when='state != "run"'),
                        ),
                        HGroup(
                            Item("save_time_trace", enabled_when='state != "run"'),
                            Item(
                                "time_trace_path",
                                width=0.1,
                                enabled_when='state != "run"',
                            ),
                        ),
                        show_border=True,
                    ),
                    Group(
                        Item("histogram", style="custom", show_label=False),
                        label="Histogram",
                        show_border=True,
                    ),
                ),
            ),
        ),
        menubar=MenuBar(
            Menu(
                Action(action="load", name="Load"),
                Action(action="save", name="Save (.pyd or .pys)"),
                Action(action="save_histogram", name="Save Histogram (.png)"),
                Action(action="saveAll", name="Save All (.png+.pys)"),
                Action(action="_on_close", name="Quit"),
                name="File",
            ),
        ),
        title="Polarization QND Measurement",
        width=1250,
        resizable=True,
        handler=PolarizationHandler,
    )


if __name__ == "__main__":
    pass
