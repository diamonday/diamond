"""
This file is part of pi3diamond.

pi3diamond is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pi3diamond is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with diamond. If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2009-2011 Helmut Fedder <helmut.fedder@gmail.com>
"""

import visa
import numpy
import logging

class SMIQ():
    """Provides control of SMIQ family microwave sources from Rhode und Schwarz with GPIB via visa."""
    _output_threshold = -90.0
    
    def __init__(self, visa_address='GPIB0::28'):
        self.visa_address = visa_address
        
    def _write(self, string):
        try: # if the connection is already open, this will work
            self.instr.write(string)
        except: # else we attempt to open the connection and try again
            try: # silently ignore possible exceptions raised by del
                del self.instr
            except Exception:
                pass
            self.instr = visa.instrument(self.visa_address)
            self.instr.write(string)
        
    def _ask(self, str):
        try:
            val = self.instr.ask(str)
        except:
            self.instr = visa.instrument(self.visa_address)
            val = self.instr.ask(str)
        return val

    def getPower(self):
        return float(self._ask(':POW?'))

    def onStatus(self):
        return float(self._ask(':OUTP?'))
    
    def setPower(self, power):    # setPower() means output_on     no need for fucking trigger
        if power is None or power < self._output_threshold:
            logging.getLogger().debug('SMIQ at '+str(self.visa_address)+' turning off.')
            self._write(':FREQ:MODE CW')
            self._write(':OUTP OFF')
            return
        logging.getLogger().debug('SMIQ at '+str(self.visa_address)+' setting power to '+str(power))
        if self.getPower()!=power:
            self._write(':FREQ:MODE CW')
            self._write(':POW %f' % float(power))
        if self.onStatus()==0:
            self._write(':OUTP ON')

    def getFrequency(self):
        return float(self._ask(':FREQ?'))

    def setFrequency(self, frequency):
        if self.getFrequency()!=frequency:
            self._write(':FREQ:MODE CW')
            self._write(':FREQ %e' % frequency)

    def setOutput(self, power, frequency):
        self.setPower(power)
        self.setFrequency(frequency)

    def initSweep(self, frequency, power):
        if len(frequency) != len(power):
            raise ValueError('Length mismatch between list of frequencies and list of powers.')
        self._write(':FREQ:MODE CW')   # Operating mode LIST is switched off
        self._write(':LIST:DEL:ALL')
        self._write('*WAI')
        self._write(":LIST:SEL 'ODMR'")   # Selection of a list or creation of a new list
        FreqString = ''
        for f in frequency[:-1]:
            FreqString += ' %f,' % f
        FreqString += ' %f' % frequency[-1]
        self._write(':LIST:FREQ' + FreqString)
        self._write('*WAI')
        PowerString = ''
        for p in power[:-1]:
            PowerString += ' %f,' % p
        PowerString += ' %f' % power[-1]
        self._write(':LIST:POW'  +  PowerString)
        self._write(':LIST:LEAR')   # Starts the LEARN function. All value pairs of the active list are
                                    # subsequently set by the instrument with the current additional
                                    # parameters, and the hardware setting data are stored.
        # This function must be called after every creating and
        # altering the list
        self._write(':TRIG1:LIST:SOUR EXT')
        # we switch frequency on negative edge. Thus, the first square pulse of the train
        # is first used for gated count and then the frequency is increased. In this way
        # the first frequency in the list will correspond exactly to the first acquired count. 
        self._write(':TRIG1:SLOP NEG') 
        self._write(':LIST:MODE STEP')   # Step-by-step manual processing of the list.
        self._write(':FREQ:MODE LIST')
        self._write('*WAI')
        N = int(numpy.round(float(self._ask(':LIST:FREQ:POIN?'))))
        if N != len(frequency):
            raise RuntimeError, 'Error in SMIQ with List Mode'

    def resetListPos(self):
        self._write(':ABOR:LIST')
        self._write('*WAI')

