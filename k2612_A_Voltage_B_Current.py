import visa
import numpy as np
from traits.api import HasTraits, Button, Range, Float
from traitsui.api import View, Item, HGroup, VGroup


class Keithley2612(HasTraits):
    Applypara = Button(label="apply parameters")
    # step = Range(low=-1e3, high=1e3, value=10, label='current step [mA]',desc='current step [mA]', auto_set=False,enter_set=True)
    # change = Button(label="change", desc="change the current by one step")
    current = Range(
        low=-1500,
        high=1500,
        value=0,
        desc="current[mA]",
        label="current[mA]",
        auto_set=False,
        enter_set=True,
        mode="slider",
    )
    voltage = Range(
        low=0,
        high=15000,
        value=0,
        desc="voltage[mV]",
        label="voltage[mV]",
        auto_set=False,
        enter_set=True,
        mode="slider",
    )
    OutputOn_A = Button(label="output on")
    OutputOff_A = Button(label="output off")
    OutputOn_B = Button(label="output on")
    OutputOff_B = Button(label="output off")
    Measure_A = Button(label="Measure")
    Measure_B = Button(label="Measure")
    ReadingV_A = Float(value=0, desc="voltage reading", label="current reading [mV]")
    ReadingI_A = Float(value=0, desc="current reading", label="current reading [mA]")
    ReadingV_B = Float(value=0, desc="voltage reading", label="current reading [mV]")
    ReadingI_B = Float(value=0, desc="current reading", label="current reading [mA]")

    # TTLOn = Button(label="TTL on")
    # TTLOff = Button(label = "TTL off")

    def __init__(self):
        self.Connect()
        self.Applypara()
        self.k2612.write("smua.source.output = smua.OUTPUT_OFF")
        self.k2612.write("smub.source.output = smub.OUTPUT_OFF")

    def Connect(self):
        self.k2612 = visa.instrument("GPIB1::26::INSTR")
        print((self.k2612.ask("*IDN?")))

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
        self.k2612.write("smua.reset()")
        self.k2612.write("smua.source.func = smua.OUTPUT_DCVOLTS")
        self.k2612.write("smua.source.autorangei = smua.AUTORANGE_ON")
        self.k2612.write("smua.source.autorangev = smua.AUTORANGE_ON")
        self.k2612.write("smua.source.levelv = " + str(np.float64(1) / 1000.0))
        self.k2612.write("smua.measure.autorangev = smua.AUTORANGE_ON")

        self.k2612.write("smub.reset()")
        self.k2612.write("smub.source.func = smub.OUTPUT_DCAMPS")
        self.k2612.write("smub.source.autorangei = smub.AUTORANGE_ON")
        self.k2612.write("smub.source.autorangev = smub.AUTORANGE_ON")
        self.k2612.write("smub.source.leveli = " + str(np.float64(1) / 1000.0))
        self.k2612.write("smub.measure.autorangei = smub.AUTORANGE_ON")

    # def _current_fired(self):
    def _OutputOn_A_fired(self):
        self.k2612.write("smua.source.output = smua.OUTPUT_ON")
        self.k2612.write(
            "smua.source.levelv = " + str(np.float64(self.voltage) / 1000.0)
        )
        # self.ReadingV = np.float(self.k2612.ask("print(smua.measure.v())")) * 1000.
        # self.ReadingI = np.float(self.k2612.ask("print(smua.measure.i())")) * 1000.

    def _OutputOff_A_fired(self):
        self.k2612.write("smua.source.output = smub.OUTPUT_OFF")

    def _OutputOn_B_fired(self):
        self.k2612.write("smub.source.output = smub.OUTPUT_ON")
        self.k2612.write(
            "smub.source.leveli = " + str(np.float64(self.current) / 1000.0)
        )
        # self.ReadingV = np.float(self.k2612.ask("print(smua.measure.v())")) * 1000.
        # self.ReadingI = np.float(self.k2612.ask("print(smua.measure.i())")) * 1000.

    def _OutputOff_B_fired(self):
        self.k2612.write("smub.source.output = smub.OUTPUT_OFF")

    def _Measure_A_fired(self):
        self.ReadingV_A = np.float(self.k2612.ask("print(smua.measure.v())")) * 1000.0
        self.ReadingI_A = np.float(self.k2612.ask("print(smua.measure.i())")) * 1000.0

    def _Measure_B_fired(self):
        self.ReadingV_B = np.float(self.k2612.ask("print(smub.measure.v())")) * 1000.0
        self.ReadingI_B = np.float(self.k2612.ask("print(smub.measure.i())")) * 1000.0

    # def _TTLOff_fired(self):
    # self.k2612.write("smua.source.output = smua.OUTPUT_OFF")

    # def _TTLOn_fired(self):
    # self.k2612.write("smua.source.output = smua.OUTPUT_ON")
    # self.k2612.write("smua.source.leveli = " + str(np.float64(1) / 1000.0))
    # self.ReadingV = np.float(self.k2612.ask("print(smua.measure.v())")) * 1000.
    # self.ReadingI = np.float(self.k2612.ask("print(smua.measure.i())")) * 1000.

    traits_view = View(
        VGroup(
            HGroup(Item("voltage", width=-500)),
            HGroup(Item("OutputOn_A"), Item("OutputOff_A")),
            HGroup(
                Item("Measure_A"),
                Item("ReadingV_A", style="readonly", width=-60),
                Item("ReadingI_A", style="readonly", width=-60),
            ),
            HGroup(Item("current", width=-500)),
            HGroup(Item("OutputOn_B"), Item("OutputOff_B")),
            HGroup(
                Item("Measure_B"),
                Item("ReadingV_B", style="readonly", width=-60),
                Item("ReadingI_B", style="readonly", width=-60),
            ),
        ),
        title="Keithley 2612 console(current mode)",
        width=600,
        height=300,
        buttons=[],
        resizable=True,
    )


def main():
    k2612 = Keithley2612()
    k2612.configure_traits()


if __name__ == "__main__":
    main()
