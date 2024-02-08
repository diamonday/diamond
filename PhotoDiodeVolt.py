from traits.api import Array, Range, Bool, Instance, on_trait_change, Enum
from traitsui.api import (
    View,
    Item,
    HGroup,
    VGroup,
)
from tools.utility import GetSetItemsHandler, GetSetItemsMixin
from tools.emod import FreeJob
from chaco.api import Plot, ArrayPlotData, hot
from hardware import nidaqmx
from enable.api import ComponentEditor
import numpy
import threading


class StartThreadHandler(GetSetItemsHandler):
    def init(self, info):
        info.object.start()


class PhotonTimeTrace(FreeJob, GetSetItemsMixin):

    TraceLength = Range(
        low=1, high=1000, value=300, desc="Length of Count Trace", label="Trace Length"
    )
    SamplesPerChannel = Range(
        low=1,
        high=10,
        value=2,
        desc="Number of Samples per channel",
        label="Samples per channel",
    )
    RefreshRate = Range(
        low=0.01, high=1, value=0.02, desc="Refresh rate [s]", label="Refresh rate [s]"
    )
    digi_channel = Enum(
        "cha0", "cha1", "cha0+1", desc="Digi channel", label="Digi channel"
    )

    # trace data
    C0 = Array()
    C1 = Array()
    C0C1 = Array()
    T = Array()

    c_enable0 = Bool(True, label="channel 0", desc="enable channel 0")
    c_enable1 = Bool(False, label="channel 1", desc="enable channel 1")
    sum_enable = Bool(False, label="c0 + c1", desc="enable sum c0 + c1")

    TracePlot = Instance(Plot)
    TraceData = Instance(ArrayPlotData)

    digits_data = Instance(ArrayPlotData)
    digits_plot = Instance(Plot)

    def __init__(self):

        super(PhotonTimeTrace, self).__init__()
        self.on_trait_change(self._update_T, "T", dispatch="ui")
        self.on_trait_change(self._update_C0, "C0", dispatch="ui")
        self.on_trait_change(self._update_C1, "C1", dispatch="ui")
        self.on_trait_change(self._update_C0C1, "C0C1", dispatch="ui")
        self._create_counter()
        self._create_digits_plot()

    def _create_counter(self):
        # self._counter0 = TimeTagger.Counter(0, int(self.SecondsPerPoint*1e12), self.TraceLength) # What's this type shape?
        # self._counter1 = TimeTagger.Counter(1, int(self.SecondsPerPoint*1e12), self.TraceLength)
        self._counter = self.Volt_Ch(
            int(self.TraceLength), int(self.SamplesPerChannel), self.C0, self.C1
        )

    def Volt_Ch(self, tl, spp, data_C0, data_C1):
        self.getdata = np.zeros((2, tl))
        self.getdata[0, :-1] = data_C0[1:]
        self.getdata[1, :-1] = data_C1[1:]
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan("Dev1/ai0:1")
            self.getdata[:, -1:] = (
                (
                    np.array(task.read(number_of_samples_per_channel=spp)).sum(
                        axis=1, keepdims=True
                    )
                )
                / spp
                + 1.381
            ) * 1000

    def _create_digits_plot(self):
        data = ArrayPlotData(image=numpy.zeros((2, 2)))
        plot = Plot(
            data,
            width=500,
            height=500,
            resizable="hv",
            aspect_ratio=37.0 / 9.0,
            padding=8,
            padding_left=48,
            padding_bottom=36,
        )
        plot.img_plot("image", xbounds=(0, 1), ybounds=(0, 1), colormap=hot)
        plot.plots["plot0"][0].value_range.high_setting = 1
        plot.plots["plot0"][0].value_range.low_setting = 0
        plot.x_axis = None
        plot.y_axis = None
        self.digits_data = data
        self.digits_plot = plot

    def _C0_default(self):
        return numpy.zeros((self.TraceLength))

    def _C1_default(self):
        return numpy.zeros((self.TraceLength))

    def _C0C1_default(self):
        return numpy.zeros((self.TraceLength))

    def _T_default(self):
        return self.RefreshRate * numpy.arange(self.TraceLength)

    def _update_T(self):
        self.TraceData.set_data("t", self.T)

    def _update_C0(self):
        self.TraceData.set_data("y0", self.C0)
        # self.TracePlot.request_redraw()
        if self.digi_channel == "cha0":
            # self.update_digits_plot(self.C0[-1])
            # averagelength=int(self.RefreshRate/self.SecondsPerPoint)
            averagelength = int(1.0 / self.RefreshRate)
            self.update_digits_plot(numpy.average(self.C0[-averagelength:]))

    def _update_C1(self):
        self.TraceData.set_data("y1", self.C1)
        # self.TracePlot.request_redraw()
        if self.digi_channel == "cha0":
            # self.update_digits_plot(self.C0[-1])
            int(1.0 / self.RefreshRate)
            # self.update_digits_plot(numpy.average(self.C1[-averagelength:]))

    def _update_C0C1(self):
        self.TraceData.set_data("y8", self.C0C1)
        # self.TracePlot.request_redraw()
        if self.digi_channel == "cha0+1":
            # self.update_digits_plot(self.C0[-1])
            int(1.0 / self.RefreshRate)
            # self.update_digits_plot(numpy.average(self.C0C1[-averagelength:]))

    def update_digits_plot(self, counts):
        # if counts>1.0e5:
        #     string = ('%5.1f' % (counts/1000.0))[:5] + 'k'
        #     data = numpy.zeros((37,9))
        #     for i, char in enumerate(string):
        #         data[6*i+1:6*i+6,1::-1] = DIGIT[char].transpose()
        #     if(counts/1000.0) >= 2e3:
        #         data *= 0.4
        # else:
        string = ("%2.3f" % counts)[:6]
        data = numpy.zeros((37, 9))
        for i, char in enumerate(string):
            data[6 * i + 1 : 6 * i + 6, 1:-1] = DIGIT[char].transpose()

        self.digits_data.set_data("image", data.transpose()[::-1])

    def _TraceLength_changed(self):
        self.C0 = self._C0_default()
        self.C1 = self._C1_default()
        self.C0C1 = self._C0C1_default()
        self.T = self._T_default()
        self._create_counter()

    def _SamplesPerChannel_changed(self):
        self.T = self._T_default()
        self._create_counter()

    def _TraceData_default(self):
        return ArrayPlotData(t=self.T, y0=self.C0, y1=self.C1, y8=self.C0C1)

    def _TracePlot_default(self):
        plot = Plot(self.TraceData, width=500, height=500, resizable="hv")
        plot.plot(("t", "y0"), type="line", color="black")
        return plot

    @on_trait_change(
        "c_enable0,c_enable1,c_enable2,c_enable3,c_enable4,c_enable5,c_enable6,sum_enable"
    )
    # when 'c_enable0'(object) change, run the function.
    def _replot(self):

        self.TracePlot = Plot(self.TraceData, width=500, height=500, resizable="hv")
        self.TracePlot.legend.align = "ll"

        n = 0
        if self.c_enable0:
            self.TracePlot.plot(
                ("t", "y0"), type="line", color="blue", name="channel 0"
            )
            n += 1
        if self.c_enable1:
            self.TracePlot.plot(("t", "y1"), type="line", color="red", name="channel 1")
            n += 1
        if self.sum_enable:
            self.TracePlot.plot(
                ("t", "y8"), type="line", color="black", name="sum c0 + c1"
            )
            n += 1

        if n > 1:
            self.TracePlot.legend.visible = True
        else:
            self.TracePlot.legend.visible = False

    def _run(self):
        """Acquire Count Trace"""
        while True:
            threading.current_thread().stop_request.wait(self.RefreshRate)
            if threading.current_thread().stop_request.isSet():
                break
            self._counter = self.Volt_Ch(
                self.TraceLength, self.SamplesPerChannel, self.C0, self.C1
            )
            self.C0 = self._counter.getdata[0]
            self.C1 = self._counter.getdata[1]
            self.C0C1 = self.C0 + self.C1

    traits_view = View(
        HGroup(
            VGroup(
                Item("TracePlot", editor=ComponentEditor(), show_label=False),
                Item("digits_plot", editor=ComponentEditor(), label="mV"),
            ),
            VGroup(Item("c_enable0"), Item("c_enable1"), Item("sum_enable")),
        ),
        Item("TraceLength"),
        Item("SamplesPerChannel"),
        Item("RefreshRate"),
        title="PDA Voltage",
        width=800,
        height=600,
        buttons=[],
        resizable=True,
        handler=StartThreadHandler,
    )


DIGIT = {}
DIGIT["-"] = numpy.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
).reshape(7, 5)
DIGIT["0"] = numpy.array(
    [
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
    ]
).reshape(7, 5)
DIGIT["1"] = numpy.array(
    [
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
    ]
).reshape(7, 5)
DIGIT["2"] = numpy.array(
    [
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
    ]
).reshape(7, 5)
DIGIT["3"] = numpy.array(
    [
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
    ]
).reshape(7, 5)
DIGIT["4"] = numpy.array(
    [
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
    ]
).reshape(7, 5)
DIGIT["5"] = numpy.array(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
    ]
).reshape(7, 5)
DIGIT["6"] = numpy.array(
    [
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
    ]
).reshape(7, 5)
DIGIT["7"] = numpy.array(
    [
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    ]
).reshape(7, 5)
DIGIT["8"] = numpy.array(
    [
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
    ]
).reshape(7, 5)
DIGIT["9"] = numpy.array(
    [
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
    ]
).reshape(7, 5)
DIGIT["."] = numpy.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
    ]
).reshape(7, 5)
DIGIT["k"] = numpy.array(
    [
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
    ]
).reshape(7, 5)
DIGIT[" "] = numpy.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
).reshape(7, 5)

if __name__ == "__main__":

    import logging

    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info("Starting logger.")

    from tools.emod import JobManager

    JobManager().start()

    p = PhotonTimeTrace()
    p.configure_traits()
