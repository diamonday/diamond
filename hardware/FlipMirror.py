import numpy as np
import time
from .nidaq import DOTask


class FlipMirror:
    def __init__(self, channel):
        self.DO_task = DOTask(channel)
        # we use a counter here instead of Digital Output channel

    def flip(self):
        # output a single down edge to flip the mirror
        self.DO_task.Write(np.array([1], dtype=np.uint8))
        # here "1" means high, 5 V
        time.sleep(0.01)
        self.DO_task.Write(np.array([0], dtype=np.uint8))
        time.sleep(0.01)

    def high(self):
        self.DO_task.Write(np.array([1], dtype=np.uint8))
        time.sleep(0.01)

    def low(self):
        self.DO_task.Write(np.array([0], dtype=np.uint8))
        time.sleep(0.01)
