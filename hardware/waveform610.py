import numpy as np
import matplotlib.pyplot as plt

from threading import Thread

from io import StringIO
from copy import deepcopy

import time

# changed from waveform.py
# change all 256 to 512
# change 4 to 8 in class Waveform->init->things about stub
# TODO: RF-Waveform
#       Waveform: Remove inheritence from Thread.
#                 Instead create thread as member.
#                 => Waveforms can be pickled
#       IQWaveform: Easy access to Waveform members:
#                   e.g. iq_wave['I'].duration => iq_wave.duration


# _____________________________________________________________________________
# PULSES:


class Pulse(object):

    def __init__(self, duration, amp=1.0, marker1=False, marker2=False):
        self.duration = int(duration)
        self.amp = amp
        self.marker1 = marker1
        self.marker2 = marker2

    # _________________
    # Operators on func

    def __add__(self, pulse):
        new = deepcopy(self)
        f = new.func
        new.func = lambda x: f(x) + pulse.func(x)
        return new

    def __iadd__(self, pulse):
        f = pulse.func
        g = self.func
        self.func = lambda x: g(x) + f(x)
        return self

    def __sub__(self, pulse):
        new = deepcopy(self)
        f = new.func
        new.func = lambda x: f(x) - pulse.func(x)
        return new

    def __isub__(self, pulse):
        f = pulse.func
        g = self.func
        self.func = lambda x: g(x) - f(x)
        return self

    def __mul__(self, pulse):
        new = deepcopy(self)
        f = new.func
        new.func = lambda x: f(x) * pulse.func(x)
        return new

    def __imul__(self, pulse):
        f = pulse.func
        g = self.func
        self.func = lambda x: g(x) * f(x)
        return self

    def __div__(self, pulse):
        new = deepcopy(self)
        f = new.func
        new.func = lambda x: f(x) / pulse.func(x)
        return new

    def __idiv__(self, pulse):
        f = pulse.func
        g = self.func
        self.func = lambda x: g(x) / f(x)
        return self

    def __neg__(self):
        new = deepcopy(self)
        f = new.func
        new.func = lambda x: -1 * f(x)
        return new

    # _____________________
    # Operators on duration

    def __lt__(self, pulse):
        """Compare durations."""
        return self.duration < pulse.duration

    def __le__(self, pulse):
        """Compare durations."""
        return self.duration <= pulse.duration

    def __gt__(self, pulse):
        """Compare durations."""
        return self.duration > pulse.duration

    def __ge__(self, pulse):
        """Compare durations."""
        return self.duration >= pulse.duration

    def __mod__(self, pulse):
        """Compare durations."""
        return self.duration % pulse.duration

    # _________
    # Function

    def compile(self, t_0):
        """This will be called by Waveform when compiling."""
        samples = np.arange(
            t_0, t_0 + self.duration, dtype=np.float64
        )  # use 8 byte precision for compilation
        samples = self.func(samples)
        samples = self.norm(samples)
        return samples

    def func(self, samples):
        """Override this to manipulate samples."""
        return samples

    def before(self, f):
        """Make a composed function f(g(t))."""
        g = self.func  # prevent recursion
        self.func = lambda t: f(g(t))

    def after(self, f):
        """Make a composed function g(f(t))."""
        g = self.func  # prevent recursion
        self.func = lambda t: g(f(t))

    def norm(self, samples):
        if len(samples) == 0:
            return samples
        max_sample = max(abs(samples))
        if max_sample > 1.0:
            samples = self.amp / max_sample * samples
        return samples


class Idle(Pulse):

    def func(self, samples):
        samples = 0.0 * samples
        return samples


class DC(Pulse):

    def func(self, samples):
        samples = 0.0 * samples + 1.0
        return samples


class Ramp(Pulse):

    def __init__(self, duration, start, stop, marker1=False, marker2=False):
        Pulse.__init__(self, duration, marker1, marker2)
        self.start = start
        self.slope = (1.0 * stop - start) / duration

    def func(self, samples):
        samples = self.start + self.slope * (samples - samples[0])
        return samples


class Envelop(Pulse):

    def __init__(self, duration, amp_li, freq_li, phase_li, amp=1.0, marker=0b00):
        Pulse.__init__(self, duration, amp, marker1=marker)
        self.amp_li = amp_li
        self.freq_li = freq_li
        self.phase_li = phase_li

    def fourier(self, segment):
        phase = 2 * np.pi * self.phase_li
        ft = self.amp_li[:, None] * np.cos(
            2 * np.pi * self.freq_li[:, None] * segment + phase[:, None]
        )
        return ft.sum(axis=0)

    def func(self, samples):
        ft = self.fourier(samples)
        return ft


class Sin(Pulse):

    def __init__(
        self, duration, freq, phase=0.0, amp=1.0, marker1=False, marker2=False
    ):
        Pulse.__init__(self, duration, amp, marker1, marker2)
        self.freq = freq
        self.phase = phase

    def func(self, samples):
        samples = self.amp * np.sin(2 * np.pi * self.freq * samples + self.phase)
        return samples


class Cos(Sin):

    def func(self, samples):
        samples = self.amp * np.cos(2 * np.pi * self.freq * samples + self.phase)
        return samples


class Gauss(Pulse):

    def __init__(
        self, duration, peak=None, sigma=None, amp=1.0, marker1=False, marker2=False
    ):
        Pulse.__init__(self, duration, amp, marker1, marker2)
        if peak is None:
            self.peak = int(duration / 2)
        else:
            self.peak = peak
        if sigma is None:
            self.sigma = int(duration / 4)
        else:
            self.sigma = sigma

    def func(self, samples):
        samples -= samples[0]  # make gauss pulse time-invariant
        samples = self.amp * np.exp(-(((samples - (self.peak)) / 2 / self.sigma) ** 2))
        return samples


class HermiteEnve(Pulse):

    def __init__(
        self, duration, c_para=0.667, T_para=0.1667, mu=0.5, amp=1.0, marker=0b00
    ):
        Pulse.__init__(self, duration, amp, marker)
        self.c_para = c_para
        self.mu = mu
        self.T_para = T_para

    def func(self, samples):
        self.peak = int(self.duration * self.mu)
        self.T_width = int(self.duration * self.T_para)
        # print(self.peak)
        samples -= samples[0]  # make envelope time-invariant
        samples = (
            self.amp
            * (1 - self.c_para * ((samples - (self.peak)) / self.T_width) ** 2)
            * np.exp(-(((samples - (self.peak)) / self.T_width) ** 2))
        )
        return samples


# _____________________________________________________________________________
# WAVEFORMS:


class Waveform610(Thread):
    """Compile a waveform from a sequence of Pulse objects upon instantiation.
    The wave form can be created as AWG-ready waveform file (*.WFM; 5 byte
    data width) or as pattern file (*.PAT; 2 byte width). Either file type
    can be handeled similar to an open file, i.e. be transmitted using
    ftplib.FTP.storbinary.

    Constructor arguments:
    Waveform(name, pulse_seq[, offset[, sampling[, file_type]]])

    name        - string - file name, extension will be added automatically
    pulse_seq   - iterable of Pulse objects - sequence of pulses from which
                                            waveform will be compiled
    [offset]    - int - time offset [samples]. default 0
    [sampling]  - float - sampling frequency [Hz]. default 1.0E+09
    [file_type] - 0 or 1 - waveform or pattern respectively.
                           default pattern
    """

    def __init__(
        self,
        name,
        pulse_seq,
        offset=0,
        file_type="PAT",
        sampling=1e9,
    ):
        self.state = "compiling"
        Thread.__init__(self)

        if file_type in [0, "WFM"]:
            self.file_type = 0
            self.name = name + ".WFM"
        elif file_type in [1, "PAT"]:
            self.file_type = 1
            self.name = name + ".PAT"
        else:
            self.file_type = 1
            self.name = name + ".PAT"

        self.index = 0
        self.pulse_seq = deepcopy(pulse_seq)
        if isinstance(self.pulse_seq, Pulse):
            self.pulse_seq = [self.pulse_seq]
        self.offset = offset
        self.sampling = sampling

        d = sum([int(p.duration) for p in self.pulse_seq])
        if d < 512:
            self.stub = 512 - d
        else:
            self.stub = (8 - d) % 8
        self.duration = d + self.stub

        self.start()

    def run(self):
        # generate header
        if self.file_type == 0:
            num_bytes = str(int(5 * self.duration))
            num_digits = str(len(num_bytes))
            header = "MAGIC 1000\r\n#" + num_digits + num_bytes
        elif self.file_type == 1:
            num_bytes = str(int(2 * self.duration))
            num_digits = str(len(num_bytes))
            header = "MAGIC 2000\r\n#" + num_digits + num_bytes

        # generate trailer
        trailer = "CLOCK %16.10e" % self.sampling

        # generate body (data)
        wave = np.zeros(
            self.duration, dtype=np.float64
        )  # use 8 byte precision for compilation
        marker1_seq = np.zeros(self.duration, dtype=np.int8)
        marker2_seq = np.zeros(self.duration, dtype=np.int8)
        t_0 = self.offset
        i = 0
        for pulse in self.pulse_seq:
            pulse.duration = int(pulse.duration)
            wave[i : i + pulse.duration] = pulse.compile(t_0)
            marker1_seq[i : i + pulse.duration] = [
                pulse.marker1,
            ] * pulse.duration
            marker2_seq[i : i + pulse.duration] = [
                pulse.marker2,
            ] * pulse.duration
            t_0 += pulse.duration
            i += pulse.duration

        # store data
        self.wave = wave
        self.marker1 = marker1_seq
        self.marker2 = marker2_seq

        # convert all to AWG-ready buffer
        # if waveform file:
        if self.file_type == 0:
            # create 5 byte data type
            # => 4 byte floating point data for waveform samples and 1 byte for markers
            dt = np.dtype([("w", "<f4"), ("m", "|i1")])

            # convert wave to 32 bit floating point data
            wave = np.float32(wave)
            # encode markers on first two LSB of a byte
            marker = marker1_seq + (marker2_seq << 1)

            # fill array with data
            data = np.array(list(zip(wave, marker)), dtype=dt)

        # if pattern file:
        elif self.file_type == 1:
            # convert wave to 10 bit DAC channel data (encoded on 10 LSB of 16 bit int)
            wave = np.int16(np.round((self.wave + 1) * 511.5))
            data = np.zeros(self.duration, dtype=np.int16)
            # encode marker1 on bit 13 and marker2 on bit 14
            data = (
                wave
                | ((marker1_seq.astype(np.int16) & 1) << 13)
                | ((marker2_seq.astype(np.int16) & 1) << 14)
            )

        # convert to read-only buffer with header and trailer
        self.data = buffer(header + data.data[:] + trailer)

        # finish thread
        self.state = "ready"

    # ____________
    # file-like; ftplib interface

    def read(self, bytes=None):
        if bytes is not None:
            r = self.data[self.index : self.index + bytes]
            self.index += bytes
        else:
            r = self.data[self.index :]
            self.index = len(self.data)
        return r

    def seek(self, byte):
        self.index = byte

    def close(self):
        pass

    # ____________
    # user interface

    def __repr__(self):
        string = "<awg520.Waveform> " + self.name
        return string

    def __str__(self):
        self.seek(0)
        return self.read()

    def save(self, filename=None, mode="b"):
        """Save waveform in CWD."""
        if filename is None:
            filename = self.name
        if mode == "b":
            self.seek(0)
            with open(filename, "w") as wfm:
                wfm.write(self.read())
        elif mode == "a":
            filename = self.name[:-4] + ".DAT"
            with open(filename, "w") as dat:
                for i in range(self.duration):
                    sample = str(self.wave[i])
                    marker1 = str(self.marker1[i])
                    marker2 = str(self.marker2[i])
                    # write
                    line = " ".join([sample, marker1, marker2]) + "\n"
                    dat.write(line)

    def plot(self):
        """Create a simple plot.

        Do not use this in a running enthought application. The wx-app of
        matplotlib will cause enthought to crash.
        """
        i = 0
        for p in self.pulse_seq:
            i += p.duration
            plt.plot([i, i], [-1, 1], "r--")
        plt.step(np.arange(len(self.wave)) + 1, self.wave, "k")
        plt.xlim((0, len(self.wave)))
        plt.ylim((-1, 1))
        plt.plot((0, len(self.wave)), (0, 0), "k--")
        plt.show()


class IQWaveform:
    """Create Waveforms for an IQ-mixer.

    Instantiating this will create two Waveform objects from the same pulse
    sequence. All Sin or Cos objects will have an additional pi/2 phase in
    the second Waveform.

    The individual Waveform objects can be accessed via attributes
    foo.x and foo.y or via foo.__getitem__(index), i.e. foo[index], where
    index may be 'i', 'I', 0 or 'q', 'Q', 1 for the
    respective Waveform object.
    """

    def __init__(self, name, pulse_seq, offset=0, sampling=1.0e09, file_type=1):
        name_i = name + "_I"
        name_q = name + "_Q"
        self.i = Waveform(
            name_i, pulse_seq, offset=offset, file_type=file_type, sampling=sampling
        )
        while not self.i.state == "ready":
            pass  # TODO: remove this bottleneck
        done = []
        for p in pulse_seq:
            try:
                p.phase
            except AttributeError as e:
                continue
            if not p in done:
                p.phase += np.pi / 2
                done.append(p)

        self.q = Waveform(
            name_q, pulse_seq, offset=offset, file_type=file_type, sampling=sampling
        )
        while not self.q.state == "ready":
            pass  # TODO: remove this bottleneck
        done = []
        for p in pulse_seq:
            try:
                p.phase
            except AttributeError as e:
                continue
            if not p in done:
                p.phase -= np.pi / 2
                done.append(p)

    def __getitem__(self, index):
        if index in ("i", "I", 0):
            return self.i
        if index in ("q", "Q", 1):
            return self.q

    def __iter__(self):
        return [self.i, self.q].__iter__()


# _____________________________________________________________________________
# SEQUENCES:


class Sequence(StringIO):

    DEFAULT_PARA = {
        "table": [
            0,
        ]
        * 16,
        "logic": [
            -1,
        ]
        * 4,
        "mode": "LOGIC",
        "timing": "ASYNC",
        "strobe": 0,
    }
    DEFAULT_ENTRY = {
        "wave": [None, None],
        "repeat": 1,
        "wait": False,
        "restart": False,
        "goto": 0,
        "idle": 0,
    }

    def __init__(self, name, **kw):
        StringIO.__init__(self)

        self.name = name + ".SEQ"

        self.is_sub = False
        self.is_main = False
        self.duration = 0
        self.settings = deepcopy(self.DEFAULT_PARA)

        for key in kw:
            self.settings[key] = kw[key]
        self.seq = []

        self.stats = [0, 0]

        self.write("MAGIC 3002\r\nLINES ")
        self.update(trailer=True)

        self.seek(0)
        self.state = "ready"

    def update(self, trailer=False):
        t = time.time()
        self.seek(18)
        self.truncate()
        # write body
        self.write("%i\r\n" % len(self.seq))
        for e in self.seq:
            if not e["idle"] == 0:
                pass
            # repeat IDLE.WFM n times, join mod to following Waveform
            # repeat, mod = int(e['idle'] / 512), int(e['idle'] % 512)

            ch = ["", ""]
            for i in [0, 1]:
                if e["wave"][i] is not None:
                    ch[i] = e["wave"][i].name
            rep = e["repeat"]
            wai = e["wait"]
            res = e["restart"]
            got = e["goto"]
            entry = (ch[0], ch[1], rep, wai, res, got)
            line = '"%s", "%s", %i, %i, %i, %i\r\n' % entry
            self.write(line)
        # write trailer
        if trailer:
            self.trailer = "TABLE_JUMP %i" + 15 * ", %i" + "\r\n"
            self.trailer %= tuple(self.settings["table"])
            self.trailer += "LOGIC_JUMP %i" + 3 * ", %i" + "\r\n"
            self.trailer %= tuple(self.settings["logic"])
            self.trailer += "JUMP_MODE " + self.settings["mode"] + "\r\n"
            self.trailer += "JUMP_TIMING " + self.settings["timing"] + "\r\n"
            self.trailer += "STROBE %i" % self.settings["strobe"]
        self.write(self.trailer)
        d = time.time() - t
        self.stats[0] += 1
        self.stats[1] = (self.stats[1] * (self.stats[0] - 1) + d) / self.stats[0]

    # ____________
    # user interface

    def __repr__(self):
        string = "<awg520.Sequence> " + self.name
        return string

    def __str__(self):
        self.seek(0)
        return self.read()

    def __getitem__(self, index):
        return self.seq.__getitem__(index)

    def __iter__(self):
        return self.seq.__iter__()

    def save_seq(self, filename=None):
        if filename is None:
            if self.name.upper().endswidth(".SEQ"):
                filename = name
            else:
                filename = name + ".SEQ"
        with open(filename, "w") as seq:
            seq.write(str(self))

    def set_table(self, table):
        self.settings["table"] = table
        self.update(trailer=True)

    def set_logic(self, logic):
        self.settings["logic"] = logic
        self.update(trailer=True)

    def set_mode(self, mode):
        self.settings["mode"] = mode
        self.update(trailer=True)

    def set_timing(self, timing):
        self.settings["timing"] = timing
        self.update(trailer=True)

    def set_strobe(self, strobe):
        self.settings["strobe"] = strobe
        self.update(trailer=True)

    def set_repeat(self, index, times):
        self.seq[index]["repeat"] = times
        self.update()

    def set_wait(self, index, value):
        self.seq[index]["wait"] = value
        self.update()

    def set_restart(self, index, value):
        self.seq[index]["restart"] = value
        self.update()

    def set_goto(self, index, line):
        self.seq[index]["goto"] = line
        self.update()

    def append(self, *wave, **kw):
        for i, w in enumerate(wave):
            if isinstance(w, Sequence):
                if self.is_sub or w.is_main:
                    return  # raise SequneceError('Nesting level...')
                else:
                    self.is_main = True
                    w.is_sub = True
            if isinstance(w, IQWaveform):
                wave = (
                    list(wave[:i])
                    + [
                        w["i"],
                        w["q"],
                    ]
                    + list(wave[min(i + 1, len(wave)) :])
                )
        l = len(wave)
        entry = deepcopy(self.DEFAULT_ENTRY)
        entry["wave"][0 : min(2, l)] = wave[0 : min(2, l)]
        for key in kw:
            entry[key] = kw[key]
        self.seq.append(entry)
        self.update()

    def extend(self, sequence):
        if self.is_sub and sequence.is_main:
            return  # raise SequnceError('Nesting level...')
        if sequence.is_main:
            self.is_main = True
        self.seq.extend(sequence.seq)
        self.update()


# _____________________________________________________________________________
# SLICED WAVEFORMS: WORK IN PROGRESS DON'T USE


class SlicedPulse:
    pass


class Sliceform(Sequence):

    def __init__(
        self, name, pulse_seq, offset=0, iq=True, file_type="PAT", sampling=1.0e9
    ):

        self.base_name = name
        self.pulse_seq = deepcopy(pulse_seq)
        self.offset = offset
        self.file_type = file_type

        super(Sliceform, self).__init__(name=self.base_name)

        self.compile()

    def num_to_letter(self):
        dict(list(zip(list(range(26)), "abcdefghijklmnopqrstuvwxyz")))

    def compile(self):
        # compile Waveforms

        carry = Idle(0)
        # break up pulse_sequence on sliceable pulses

        index_0 = 0
        sliced = []
        stub = 0
        t_0 = self.offset
        for index, pulse in enumerate(pulse_seq):
            if isinstance(pulse, Idle) and pulse.duration - stub >= 512:
                if index_0 < index:
                    self.base_name + ""
                    # slice
                    if index_0 == 0:
                        piece = pulse_seq[index_0:index]
                    else:
                        piece = [
                            carry,
                        ] + pulse_seq[index_0:index]

                    # create Waveforms
                    if self.iq:
                        sliced.append(IQWaveform(piece, t_0))
                    else:
                        sliced.append(Waveform())
                # sliced.append(pulse_seq[i])
                # index_0 = index
        # if not i_0 == len(pulse_seq):
        # sliced.append(pulse_seq[i_0:])


# debug script ################################################################
if __name__ == "__main__":

    pi_half = Sin(50, 0.01)
    pi = Sin(100, 0.01)
    evo = Idle()

    seq = Sliceform("FOO", [pi_half, evo, pi, evo, pi_half])

    print(seq)
