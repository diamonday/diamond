import numpy as np
import visa
from traits.api import Button, Float, HasTraits, Range
from traitsui.api import HGroup, Item, VGroup, View


class Keithley2612(HasTraits):
    Applypara = Button(label="apply parameters")
    # step = Range(low=-1e3, high=1e3, value=10, label='current step [mA]',desc='current step [mA]', auto_set=False,enter_set=True)
    # change = Button(label="change", desc="change the current by one step")
    current = Range(
        low=0,
        high=15000,
        value=0,
        desc="current[mA]",
        label="current[mA]",
        auto_set=False,
        enter_set=True,
        mode="slider",
    )
    OutputOn = Button(label="output on")
    OutputOff = Button(label="output off")
    Measure = Button(label="Measure")
    ReadingV = Float(value=0, desc="voltage reading", label="current reading [mV]")
    ReadingI = Float(value=0, desc="current reading", label="current reading [mA]")

    TTLOn = Button(label="TTL on")
    TTLOff = Button(label="TTL off")

    def __init__(self):
        self.Connect()
        self.Applypara()
        self.k2612.write("smua.source.output = smua.OUTPUT_ON")
        self.k2612.write("smub.source.output = smub.OUTPUT_ON")

    def Connect(self):
        self.k2612 = visa.instrument("GPIB1::26::INSTR")
        print(self.k2612.ask("*IDN?"))

    def Applypara(self):
        # self.k2612.write("smua.reset()")
        # self.k2612.write("smua.source.func = smua.OUTPUT_DCAMPS")  #Selects current source function
        # self.k2612.write("smua.source.func = smuX.OUTPUT_DCVOLTS")  #Selects current source function
        # self.k2612.write("smua.source.autorangei = smua.AUTORANGE_ON")  #autorangeY  where Y->(v = current, i = current)
        # self.k2612.write("smua.source.autorangev = smua.AUTORANGE_ON")
        # self.k2612.write("smua.source.leveli = " + str(np.float64(self.current) / 1000.0))
        # self.k2612.write("smua.source.levelv = " + str(np.float64(self.current) / 1000.0))
        # self.k2612.write("smua.measure.autorangei = smua.AUTORANGE_ON")
        # self.k2612.write("smua.measure.autorangev = smua.AUTORANGE_ON")

        self.k2612.write("smub.reset()")
        self.k2612.write("smub.source.func = smub.OUTPUT_DCAMPS")
        self.k2612.write("smub.source.autorangei = smub.AUTORANGE_ON")
        self.k2612.write("smub.source.autorangev = smub.AUTORANGE_ON")
        self.k2612.write("smub.source.leveli = " + str(np.float64(1) / 1000.0))
        self.k2612.write("smub.measure.autorangei = smub.AUTORANGE_ON")

    # def _current_fired(self):

    def _OutputOn_fired(self):
        self.k2612.write("smub.source.output = smub.OUTPUT_ON")
        self.k2612.write(
            "smub.source.leveli = " + str(np.float64(self.current) / 1000.0)
        )
        # self.ReadingV = np.float(self.k2612.ask("print(smua.measure.v())")) * 1000.
        # self.ReadingI = np.float(self.k2612.ask("print(smua.measure.i())")) * 1000.

    def _OutputOff_fired(self):
        self.k2612.write("smub.source.output = smub.OUTPUT_OFF")

    def _Measure_fired(self):
        self.ReadingV = np.float(self.k2612.ask("print(smub.measure.v())")) * 1000.0
        self.ReadingI = np.float(self.k2612.ask("print(smub.measure.i())")) * 1000.0

    def _TTLOff_fired(self):
        self.k2612.write("smua.source.output = smua.OUTPUT_OFF")

    def _TTLOn_fired(self):
        self.k2612.write("smua.source.output = smua.OUTPUT_ON")
        self.k2612.write("smua.source.leveli = " + str(np.float64(1) / 1000.0))
        # self.ReadingV = np.float(self.k2612.ask("print(smua.measure.v())")) * 1000.
        # self.ReadingI = np.float(self.k2612.ask("print(smua.measure.i())")) * 1000.

    traits_view = View(
        VGroup(
            HGroup(Item("current", width=-500)),
            HGroup(Item("OutputOn"), Item("OutputOff")),
            HGroup(
                Item("Measure"),
                Item("ReadingV", style="readonly", width=-60),
                Item("ReadingI", style="readonly", width=-60),
            ),
            HGroup(Item("TTLOn"), Item("TTLOff")),
        ),
        title="Keithley 2612 console(current mode)",
        width=600,
        height=150,
        buttons=[],
        resizable=True,
    )


def main():
    k2612 = Keithley2612()
    k2612.configure_traits()


if __name__ == "__main__":
    main()
