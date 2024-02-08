from hardware.api import PulseGenerator
import numpy as np


pg_test = PulseGenerator()


def convertSequenceToBinary(sequence, loop=True):
    """
    Converts a pulse sequence (list of tuples (channels,time) )
    into a series of pulser instructions (128 bit words),
    and returns these in a binary buffer of length N*1024
    representing N SDRAM pages.

    A pulser instruction has the form

    command (1 bit) | repetition (31 bit) | ch0 pattern (8bit/4bit), ..., chN pattern (8bit/4bit)'

    The pulse sequence is split into a suitable series of
    such low level pulse commands taking into account the
    minimal 8 bit pattern length.

    input:

        sequence    list of tuples of the form (channels, time), where channels is
                    a list of strings corresponding to channel names and time is a float
                    specifying the time in ns.

        loop        if True, repeat the sequence indefinitely, if False, run the sequence once

    returns:

        buf         binary buffer containing N SDRAM pages that represent the sequence
    """

    dt = pg_test.dt
    N_CHANNELS, CHANNEL_WIDTH = pg_test.n_channels, pg_test.channel_width
    ONES = 2**CHANNEL_WIDTH - 1
    REP_MAX = 2**31
    buf = ""
    # we start with an integer zero for each channel.
    # In the following, we will start filling up the bits in each of these integers
    blank = np.zeros(
        N_CHANNELS, dtype=int
    )  # we will need this many times, so we create it once and copy from this
    pattern = blank.copy()
    index = 0
    for channels, time in sequence:
        ticks = int(
            round(time / dt)
        )  # convert the time into an integer multiple of hardware time steps
        if ticks is 0:
            continue
        bits = pg_test.createBitsFromChannels(channels)
        if (
            index + ticks < CHANNEL_WIDTH
        ):  # if pattern does not fill current block, insert into current block and continue
            pg_test.setBits(pattern, index, ticks, bits)
            index += ticks
            print(pattern)
            continue
        if (
            index > 0
        ):  # else fill current block with pattern, reduce ticks accordingly, write block and start a new block
            pg_test.setBits(pattern, index, CHANNEL_WIDTH - index, bits)
            buf += pg_test.pack(0, pattern)
            print(("bin:", [ord(s) for s in pg_test.pack(0, pattern)]))
            print(pattern)
            ticks -= CHANNEL_WIDTH - index
            pattern = blank.copy()
        # split possible remaining ticks into a command with repetitions and a single block for the remainder
        repetitions = ticks / CHANNEL_WIDTH  # number of full blocks
        index = (
            ticks % CHANNEL_WIDTH
        )  # remainder will make the beginning of a new block
        print((channels, time, repetitions, index, ticks))
        if repetitions > 0:
            if repetitions > REP_MAX:
                multiplier = repetitions / REP_MAX
                repetitions = repetitions % REP_MAX
                buf += multiplier * pg_test.pack(REP_MAX - 1, ONES * bits)
            buf += pg_test.pack(
                repetitions - 1, ONES * bits
            )  # rep=0 means the block is executed once
            print(
                ("rep:", [ord(s) for s in pg_test.pack(repetitions - 1, ONES * bits)])
            )
            print(bits)
        if index > 0:
            pattern = blank.copy()
            pg_test.setBits(pattern, 0, index, bits)
            print(pattern)
    if loop:  # repeat the hole sequence
        if index > 0:  # fill up incomplete block with zeros and write it
            pg_test.setBits(
                pattern, index, CHANNEL_WIDTH - index, np.zeros(N_CHANNELS, dtype=bool)
            )
            buf += pg_test.pack(0, pattern)
            print(("bin:", [ord(s) for s in pg_test.pack(0, pattern)]))
            print(("82:", pattern, index))
    else:  # stop after one execution
        if index > 0:  # fill up the incomplete block with the bits of the last step
            pg_test.setBits(pattern, index, CHANNEL_WIDTH - index, bits)
            buf += pg_test.pack(0, pattern)
            print(("bin:", [ord(s) for s in pg_test.pack(0, pattern)]))
            print(("88:", pattern, index))
        buf += pg_test.pack(1 << 31, ONES * bits)
        buf += pg_test.pack(1 << 31, ONES * bits)

    # print "buf has",len(buf)," bytes"
    buf = (
        buf + ((1024 - len(buf)) % 1024) * "\x00"
    )  # pad buffer with zeros so it matches SDRAM / FIFO page size
    print(("buf has", len(buf), " bytes"))
    return buf


Seq = [(["mw_y"], 1.5 * 5), ([], 1.5 * 3), (["green"], 1.5 * 40)]
bstr = convertSequenceToBinary(Seq, loop=False)
bstr_readable = [ord(s) for s in bstr]

# print(bstr_readable)
