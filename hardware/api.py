
"""
Hardware API is defined here.

Example of usage:

from hardware.api import PulseGenerator
PG = PulseGenerator  

Default hardware api hooks are dummy classes.

Provide a file 'custom_api.py' to define actual hardware API hooks.
This can be imported names, modules, factory functions and factory functions
that emulate singleton behavior.

See 'custom_api_example.py' for examples.
"""

import numpy as np
import logging
import time
import os
from pulse_generator import PulseGenerator as PulseGeneratorBase
# import the PulseGenerator class of the pulse_generator.py file as PulseGeneratorBase


class Scanner(  ):
    def getXRange(self):
        return (0.,100.)
    def getYRange(self):
        return (0.,100.)
    def getZRange(self):
        return (-20.,20.)
    def setx(self, x):
        pass
    def sety(self, y):
        pass
    def setz(self, z):
        pass
    def setPosition(self, x, y, z):
        """Move stage to x, y, z"""
        pass
    def scanLine(self, Line, SecondsPerPoint, return_speed=None):
        time.sleep(0.1)
        return (1000*np.sin(Line[0,:])*np.sin(Line[1,:])*np.exp(-Line[2,:]**2)).astype(int)


class Counter(  ):
    def configure(self, n, SecondsPerPoint, DutyCycle=0.8):
        x = np.arange(n)
        a = 100.
        c = 50.
        x0 = n/2.
        g = n/10.
        y = np.int32( c - a / np.pi * (  g**2 / ( (x-x0)**2 + g**2 )  ) )
        Counter._sweeps = 0
        Counter._y = y
    def run(self):
        time.sleep(1)
        Counter._sweeps+=1
        return np.random.poisson(Counter._sweeps*Counter._y)
    def clear(self):
        pass


class Microwave(  ):
    def setPower(self, power):
        logging.getLogger().debug('Setting microwave power to '+str(power)+'.')
    def setOutput(self, power, frequency):
        logging.getLogger().debug('Setting microwave to p='+str(power)+' f='+str(frequency)+'.')
    def initSweep(self, f, p):
        logging.getLogger().debug('Setting microwave to sweep between frequencies %e .. %e with power %f.'%(f[0],f[-1],p[0]))
    def resetListPos(self):
        pass


class PulseGeneratorClass( PulseGeneratorBase ):
        
    def Continuous(self, channels):
        self.setContinuous(channels)
        
    def Sequence(self, sequence, loop=True):
        self.setSequence(sequence, loop)
        
    def Run(self, loop=None):
        self.runSequence(loop=True)
        
    def Night(self):
        self.setContinuous(0x0000)

    def Light(self):
        self.Continuous(['flip','green'])

    def Open(self):
        self.setContinuous(0xffff)


# if customized hardware factory is present run it
# Provide this file to overwrite / add your own factory functions, classes, imports

if os.access('hardware/custom_api.py', os.F_OK):
    execfile('hardware/custom_api.py')
