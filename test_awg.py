import hardware.awg520_manager
import importlib

importlib.reload(hardware.awg520_manager)


def AWG520_test():
    from hardware.awg520_manager import AWGManager

    return AWGManager(
        gpib="GPIB0::4::INSTR", ftp="192.168.1.6", socket=("192.168.1.6", 4000)
    )


awgt = AWG520_test()
awgt.edit_traits()
