
from tools.utility import singleton


'''
for nidaq 
CTR0 OUT is connected to smiq trigger
user1->PFI3->CTR1 SRC, connected to time_tagger
ao 0,1 of connector0 connected to 2d mirror
ao 0 of connector1 connected to piezo
'''
@singleton
def Scanner():
    from nidaq import Scanner
    return Scanner( CounterIn='/Dev1/Ctr1',
                    CounterOut='/Dev1/Ctr0',
                    TickSource='/Dev1/PFI3',  
                    AOChannels='/Dev1/ao0:2',
                    x_range=(-300,300.0),
                    y_range=(-300,300.0),
                    z_range=(0,100.),
                    v_range=(0.,10.))      # the voltage range here is only used to give a restriction to the Multiboard class in nidaq.py


@singleton
def Counter():
    from nidaq import PulseTrainCounter
    return PulseTrainCounter(CounterIn='/Dev1/Ctr3',    # Why here is CTR 3 not CTR 1?
                              CounterOut='/Dev1/Ctr0',          
                              TickSource='/Dev1/PFI3')   # CTR 1 SRC

def CountTask(bin_width, length):
    return  TimeTagger.Counter(0,int(bin_width*1e12), length)
# what's this for?


def FlipMirror():
    from FlipMirror import FlipMirror
    flip_mirror = FlipMirror('/Dev1/port0/line7')
    return flip_mirror



@singleton
def Microwave():
    import microwave_sources
    return microwave_sources.SMIQ(visa_address='GPIB0::7')   # the one below awg610 on the shelf

@singleton
def MicrowaveB():
    import microwave_sources
    return microwave_sources.SMIQ(visa_address='GPIB0::29')  # the one at the bottom of shelf

@singleton
def MicrowaveC():
    import microwave_sources
    return microwave_sources.SMIQ(visa_address='GPIB0::27')  # the one on the shelf above table

MicrowaveA = Microwave

@singleton
def RFSource():
    import rf_source
    return rf_source.Rigol822(visa_address='USB0::0x1AB1::0x643::DG8A241801155::INSTR')  #address unsure

@singleton
def AWG520():
    from hardware.awg520_manager import AWGManager
    return AWGManager(
        gpib='GPIB0::4::INSTR',
        ftp='192.168.1.6',
        socket=('192.168.1.6',4000)
    )

#def AWG520():
#    from awg520 import AWGHasTraits
#    return AWGHasTraits(
#                        gpib='GPIB0::4::INSTR',
#                        ftp='192.168.1.6',
#                        socket=('192.168.1.6',4000))#address unsure since socket seems not used
    
    
#@singleton
#def AWG610():
#    from awg610 import AWGHasTraits as AWGHasTraits610
#    return AWGHasTraits610(
#                        gpib='GPIB0::1::INSTR',
#                        ftp='192.168.1.7',
#                        socket=('192.168.1.7',4000))#address unsure since socket seems not used
                        
                            
@singleton
def AWG610():
    from hardware.awg610_manager import AWGManager610
    return AWGManager610(
                        gpib='GPIB0::1::INSTR',
                        ftp='192.168.1.7',
                        socket=('192.168.1.7',4000))#address unsure since socket seems not used
                        
                        
@singleton
def PulseGenerator():
    channel_map = {
        'awg610':0, 'awg_rf':0,
        'awg520':1, 'awg':1, 'flip':1,
        'ch2':2,
        'ch3':3,# 'green':3, #'aom':3,
        'ch4':4,
        'ch5':5,'aom':5, 'green':5,
        'laser':6,
        'sequence':7,
        'ch8':8, 'A':8, 'mw_13':8,
        'ch9':9, 'B':9, 'mw_24':9,
        'ch10':10, 'C':10, 'mw_y':10, 'mw_all_y':10,
        'ch11':11, 'D':11, 'mw_x':11, 'mw_all_x':11,
    }
    return PulseGeneratorClass(serial='1634000FWM',channel_map=channel_map)


import TimeTagger as tt
tagger=tt.TimeTagger('1729000IB0')
#tagger=tt.TimeTagger('2138000XGM') #TT from qichao group
# import the TimeTagger.py file, there is a TimeTagger class inside it

import time_tagger_control
TimeTagger=time_tagger_control.TimeTaggerControl(tagger)
# import the time_tagger_control.py file, there is a TimeTaggerControl class inside it.
# used to constuct the user interface to control the device

