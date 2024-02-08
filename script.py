'''
import sys
from measurements.polarization import Polarization_QND

class redirect_output(object):
    """context manager for reditrecting stdout/err to files"""


    def __init__(self, stdout='', stderr=''):
        self.stdout = stdout
        self.stderr = stderr

    def __enter__(self):
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr

        if self.stdout:
            sys.stdout = open(self.stdout, 'w')
        if self.stderr:
            if self.stderr == self.stdout:
                sys.stderr = sys.stdout
            else:
                sys.stderr = open(self.stderr, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr
        

with redirect_output(stderr="my_err.txt"):
    Polarization_QND().edit_traits()
'''

import hardware.api as ha
import hardware
import measurements

import measurements.poisson
reload(measurements.poisson)

from measurements.poisson import Poisson
poisson = Poisson()
poisson.edit_traits()

'''
awg = ha.AWG520()
awg.edit_traits()
from measurements.pulsed_awg import Rabi
rabiawg=Rabi(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520())
rabiawg.edit_traits()
#import measurements.pulsed_awg
#reload(measurements.pulsed_awg)

import measurements.odmr_auto_record
reload(measurements.odmr_auto_record)
from  measurements.odmr_auto_record import ODMR as aODMR2

odmr_auto2 = aODMR2()
odmr_auto2.edit_traits()
'''

'''
import measurements.correlation
reload(measurements.correlation)
from measurements.correlation import Correlation
corr=Correlation()
corr.edit_traits()




from measurements.pulsed_awg import Rabi

rabiawg = Rabi(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520())
rabiawg.edit_traits()
'''
'''
import measurements.pulsed_awg
reload(measurements.pulsed_awg)

from measurements.pulsed_awg import Count_decay

count_decay = Count_decay(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520())
count_decay.edit_traits()
'''
'''
from measurements.pulsed_awg import CPMG, CPMG_block

cpmg0 = CPMG(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520())
cpmg0.edit_traits()
#cpmg = CPMG_block(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520())
#cpmg.edit_traits()
'''
'''


from measurements.pulsed_awg import Trigger
trig = Trigger(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520())
trig.edit_traits()
'''

'''
import measurements.tomo_awg
reload(measurements.tomo_awg)
from measurements.tomo_awg import Tomo
tomo = Tomo(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520())
tomo.edit_traits()
'''
'''
import analysis.fitting
reload(analysis.fitting)

import measurements.NuclearRabi
reload(measurements.NuclearRabi)

from measurements.NuclearRabi import NuclearRabi
nuclearrabi=NuclearRabi(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
nuclearrabi.edit_traits()
'''

'''
import measurements.pulsed_awg_rf
reload(measurements.pulsed_awg_rf)

import measurements.VQE_N_smiq 
reload(measurements.VQE_N_smiq)

from measurements.VQE_N_smiq import VQE_N_smiq
vqe_n_smiq=VQE_N_smiq(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg_rf=ha.AWG610())
vqe_n_smiq.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.VQE_only_population 
reload(measurements.VQE_only_population)
from measurements.VQE_only_population import VQE
vqe=VQE(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
vqe.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.VQE_only_population_check_N 
reload(measurements.VQE_only_population_check_N)
from measurements.VQE_only_population_check_N import VQE_test
vqe_test=VQE_test(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
vqe_test.edit_traits()
'''

'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.VQE_en 
reload(measurements.VQE_en)
from measurements.VQE_en import VQE_en
vqe_en=VQE_en(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
vqe_en.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.VQE_only_population 
reload(measurements.VQE_only_population)

from measurements.VQE_only_population import VQE
vqe=VQE(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
vqe.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.VQE_N3N4_only 
reload(measurements.VQE_N3N4_only)

from measurements.VQE_N3N4_only import VQE_N3N4_only
vqe_N3N4_only=VQE_N3N4_only(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
vqe_N3N4_only.edit_traits()
'''

'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.NMR_awg
reload(measurements.NMR_awg)

from measurements.NMR_awg import NMR_awg
nmr_awg=NMR_awg(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
nmr_awg.edit_traits()
'''
'''
import measurements.pulsed_awg_rf
reload(measurements.pulsed_awg_rf)

import measurements.NMR_smiq_awg_rf
reload(measurements.NMR_smiq_awg_rf)

from measurements.NMR_smiq_awg_rf import NMR_awg
nmr_smiq_awg_rf=NMR_awg(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg_rf=ha.AWG610())
nmr_smiq_awg_rf.edit_traits()

import measurements.pulsed_awg_rf
reload(measurements.pulsed_awg_rf)

import measurements.NuclearRabi_smiq
reload(measurements.NuclearRabi_smiq)

from measurements.NuclearRabi_smiq import NuclearRabi_smiq
nuclearrabi_smiq=NuclearRabi_smiq(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg_rf=ha.AWG610())
nuclearrabi_smiq.edit_traits()
'''

'''
import measurements.pulsed_awg_rf
reload(measurements.pulsed_awg_rf)

import measurements.NuclearRabi_smiq_Nnorm
reload(measurements.NuclearRabi_smiq_Nnorm)

from measurements.NuclearRabi_smiq_Nnorm import NuclearRabi_smiq_Nnorm
nuclearrabi_smiq_nnorm=NuclearRabi_smiq_Nnorm(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg_rf=ha.AWG610())
nuclearrabi_smiq_nnorm.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.NuclearRabi_freq
reload(measurements.NuclearRabi_freq)

from measurements.NuclearRabi_freq import NuclearRabi
nuclearrabi=NuclearRabi(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
nuclearrabi.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.NODMR_awg
reload(measurements.NODMR_awg)

from measurements.NODMR_awg import NODMR_awg
nodmr_awg2=NODMR_awg(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
nodmr_awg2.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.NuclearRabi_e1
reload(measurements.NuclearRabi_e1)

from measurements.NuclearRabi_e1 import NuclearRabi_e1
nuclearrabi_e1=NuclearRabi_e1(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
nuclearrabi_e1.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.NuclearRabi_e1
reload(measurements.NuclearRabi_e1)

from measurements.NuclearRabi_e1_trig import NuclearRabi_e1_trig
nuclearrabi_e1_trig=NuclearRabi_e1_trig(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
nuclearrabi_e1_trig.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.NuclearRabi_real
reload(measurements.NuclearRabi_real)

from measurements.NuclearRabi_real import NuclearRabi_real
nuclearrabi_real=NuclearRabi_real(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
nuclearrabi_real.edit_traits()
'''
'''
import measurements.pulsed_2smiq_awg_rf
reload(measurements.pulsed_2smiq_awg_rf)

import measurements.NuclearRabi_tomo_BigPi
reload(measurements.NuclearRabi_tomo_BigPi)

from measurements.NuclearRabi_tomo_BigPi import NuclearRabi_smiq_Nnorm
nuclearrabi_smiq_nnorm=NuclearRabi_smiq_Nnorm(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), microwave2=ha.MicrowaveB(), awg_rf=ha.AWG610())
nuclearrabi_smiq_nnorm.edit_traits()
'''
'''
import measurements.pulsed_2smiq_awg_rf
reload(measurements.pulsed_2smiq_awg_rf)

import measurements.NuclearRabi_2smiq_real
reload(measurements.NuclearRabi_2smiq_real)

from measurements.NuclearRabi_2smiq_real import NuclearRabi_2smiq_real
nuclearrabi_2smiq_real=NuclearRabi_2smiq_real(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), microwave2=ha.MicrowaveB(), awg_rf=ha.AWG610())
nuclearrabi_2smiq_real.edit_traits()
'''
'''
import measurements.pulsed_2smiq_awg_rf
reload(measurements.pulsed_2smiq_awg_rf)

import measurements.Pi_test
reload(measurements.Pi_test)

from measurements.Pi_test import Pi_test
pi_test=Pi_test(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), microwave2=ha.MicrowaveB(), awg_rf=ha.AWG610())
pi_test.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.NuclearRabi_e0_rfmwrf
reload(measurements.NuclearRabi_e0_rfmwrf)

from measurements.NuclearRabi_e0_rfmwrf import NuclearRabi_e0_rfmwrf
nuclearrabi_e0_rfmwrf=NuclearRabi_e0_rfmwrf(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
nuclearrabi_e0_rfmwrf.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.NMR_e0_awg
reload(measurements.NMR_e0_awg)

from measurements.NMR_e0_awg import NMR_e0_awg
nmr_e0_awg=NMR_e0_awg(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
nmr_e0_awg.edit_traits()
'''
'''
import measurements.pulsed_2awg
reload(measurements.pulsed_2awg)

import measurements.NMR_real_awg
reload(measurements.NMR_real_awg)

from measurements.NMR_real_awg import NMR_real_awg
nmr_real_awg=NMR_real_awg(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
nmr_real_awg.edit_traits()
'''
'''
import measurements.pulsed
reload(measurements.pulsed)
import analysis.pulsed
reload(analysis.pulsed)
from analysis.pulsed import PulsedAnalyzer
pa6 = PulsedAnalyzer()
pa6.edit_traits()
'''
'''
import tools.threading_monitor
reload(tools.threading_monitor)
from tools.threading_monitor import EventMonitor
test = EventMonitor()
test.edit_traits()
'''