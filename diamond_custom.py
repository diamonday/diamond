from datetime import date
import os

if __name__ == "__main__":

    import hardware.api as ha

    # start confocal including auto_focus tool
    from measurements.confocal import Confocal

    scanner = ha.Scanner()
    confocal = Confocal(scanner)
    confocal.edit_traits()

    # the "Counter" user interface
    from measurements.photon_time_trace1 import PhotonTimeTrace

    time_trace = PhotonTimeTrace()
    time_trace.edit_traits()

    """
    default_path_autofocus = "D:\\Data\\2023\\Autofocus\\autofocus.pys"
    from measurements.auto_focus import AutoFocus
    auto_focus = AutoFocus(confocal)
    auto_focus.edit_traits()
    
    try:
        auto_focus.load(filename=default_path_autofocus)
    except:
        pass
    """

    from measurements.auto_focus_trace import AutoFocusTrace

    auto_focus_trace = AutoFocusTrace(confocal)
    auto_focus_trace.edit_traits()

    default_path_autofocus_trace = "D:\\Data\\2023\\Autofocus\\autofocus_trace.pys"

    try:
        auto_focus_trace.load(filename=default_path_autofocus_trace)
    except:
        pass

    """
    # from measurements.odmr import ODMR
    from measurements.odmr_t import ODMR
    odmr = ODMR()
    #odmr.edit_traits()
    """

    from analysis.pulsed import PulsedAnalyzer

    pulsed_analyzer = PulsedAnalyzer()
    pulsed_analyzer.edit_traits()
    pulsed_analyzer2 = PulsedAnalyzer()
    pulsed_analyzer2.edit_traits()

    pulsed_analyzer3 = PulsedAnalyzer()
    # pulsed_analyzer3.edit_traits()
    # from measurements.DDoscillation2 import DDosc
    # ddosc = DDosc()
    # ddosc.edit_traits()

    # from measurements.PulPol import PulPol
    # pulpol = PulPol()
    # pulpol.edit_traits()

    # from measurements.SNSCSXYn import SNSCXYn
    # snscxyn = SNSCXYn()
    # snscxyn2 = SNSCXYn()
    # snscxyn.edit_traits()
    # snscxyn2.edit_traits()

    # from measurements.SNSCSXYn_2d_no_cursor_new_t2 import SNSCXYn2d
    # snscxyn2d = SNSCXYn2d()
    # snscxyn2d.edit_traits()

    # from measurements.polarization import Polarization_QND
    # polar = Polarization_QND()
    # polar.edit_traits()

    """
    from measurements.NuclearRabi_smiq import NuclearRabi_smiq
    nuclearrabi_smiq=NuclearRabi_smiq(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg_rf=ha.AWG610())
    nuclearrabi_smiq.edit_traits()
    
    
    from measurements.NMR_smiq_awg_rf import NMR_awg
    nmr_smiq_awg_rf=NMR_awg(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg_rf=ha.AWG610())
    nmr_smiq_awg_rf.edit_traits()
    """

    """
    from measurements.NuclearRabi_tomo_heating import NuclearRabi_smiq_Nnorm
    nuclearrabi_smiq_nnorm=NuclearRabi_smiq_Nnorm(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), microwave2=ha.MicrowaveB(), awg_rf=ha.AWG610())
    nuclearrabi_smiq_nnorm.edit_traits()
    
    
    from measurements.ERabi_TOMO import ERabi_TOMO
    erabi_tomo=ERabi_TOMO(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), microwave2=ha.MicrowaveB(), awg_rf=ha.AWG610())
    erabi_tomo.edit_traits()
    """
    """
    from measurements.VQE_1p_N import VQE_1p_N
    vqe_1p_n=VQE_1p_N(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), microwave2=ha.MicrowaveB(), awg_rf=ha.AWG610())
    vqe_1p_n.edit_traits()
    """
    """
    awg610 = ha.AWG610()
    awg610.edit_traits()
    """

    """
    from measurements.VQE_1p_diag import VQE_1p_diag
    vqe_1p_diag=VQE_1p_diag(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), microwave2=ha.MicrowaveB(), awg_rf=ha.AWG610())
    vqe_1p_diag.edit_traits()
    """
    """
    from measurements.VQE_1p import VQE_1p
    vqe_1p=VQE_1p(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), microwave2=ha.MicrowaveB(), awg_rf=ha.AWG610())
    vqe_1p.edit_traits()
    """

    """
    from measurements.VQE_2p_4 import VQE_2p
    vqe_2p=VQE_2p(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), microwave2=ha.MicrowaveB(), microwave3=ha.MicrowaveC(), awg_rf=ha.AWG610())
    vqe_2p.edit_traits()
    """

    """
    from measurements.NuclearRabi_smiq_Nnorm_mid import NuclearRabi_smiq_Nnorm
    nuclearrabi_smiq_nnorm=NuclearRabi_smiq_Nnorm(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg_rf=ha.AWG610())
    nuclearrabi_smiq_nnorm.edit_traits()
    """
    """
    from measurements.NuclearRabi_2smiq_tomo import NuclearRabi_smiq_Nnorm
    nuclearrabi_smiq_nnorm=NuclearRabi_smiq_Nnorm(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), microwave2=ha.MicrowaveB(), awg_rf=ha.AWG610())
    nuclearrabi_smiq_nnorm.edit_traits()
    """
    """
    import measurements.pulsed_awg_rf

    import measurements.VQE_N_smiq 

    from measurements.VQE_N_smiq import VQE_N_smiq
    vqe_n_smiq=VQE_N_smiq(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg_rf=ha.AWG610())
    vqe_n_smiq.edit_traits()
    
    
    from measurements.NMR_awg import NMR_awg
    nmr_awg=NMR_awg(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
    nmr_awg.edit_traits()
    from measurements.VQE_only_population import VQE
    vqe=VQE(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
    vqe.edit_traits()

    from measurements.VQE_only_population import VQE
    vqe=VQE(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
    vqe.edit_traits()

    from measurements.NuclearRabi import NuclearRabi
    #awg_rf = ha.AWG610()
    nuclearrabi=NuclearRabi(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520(), awg_rf=ha.AWG610())
    nuclearrabi.edit_traits()
    #awg_rf.edit_traits()
    """

    """
    awg = ha.AWG520()
    awg.edit_traits()
    
    from measurements.pulsed_awg import Rabi
    rabiawg=Rabi(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520())
    rabiawg.edit_traits()
    
    from measurements.tomo_awg import Tomo

    tomo = Tomo(pulse_generator=ha.PulseGenerator(), time_tagger=ha.TimeTagger, microwave=ha.Microwave(), awg=ha.AWG520())
    tomo.edit_traits()
    """
    """
    from measurements.correlation import Correlation
    corr=Correlation()
    corr.edit_traits()
    """
    default_path_autoODMR = "D:\\Data\\2023\\AutoODMR\\ODMR.pys"
    default_path_autoODMR_AWG = "D:\\Data\\2023\\AutoODMR\\ODMR_AWG.pys"
    # from  measurements.odmr_auto import ODMR as aODMR
    from measurements.odmr_auto_record import ODMR as aODMR

    # odmr_auto = aODMR(snscxyn,snscxyn2,snscxyn2d,pulpol,rabiawg,tomo)
    # odmr_auto = aODMR(rabiawg,tomo)
    # odmr_auto = aODMR(rabiawg,tomo,nuclearrabi,vqe,pulsed_analyzer,corr)
    # odmr_auto = aODMR(pulsed_analyzer,nuclearrabi_smiq,nmr_smiq_awg_rf)
    """
    from measurements.poisson import Poisson
    poisson = Poisson()
    poisson.edit_traits()
    """

    odmr_auto = aODMR(pulsed_analyzer, pulsed_analyzer2)

    try:
        odmr_auto.load(filename=default_path_autoODMR)
    except:
        pass

    odmr_auto.edit_traits()

    # open the 520 laser
    # ha.PulseGenerator().Light()

    t = date.today().timetuple()
    if t.tm_mday < 10:
        d = "0" + str(t.tm_mday)
    else:
        d = str(t.tm_mday)
    if t.tm_mon < 10:
        m = "0" + str(t.tm_mon)
    else:
        m = str(t.tm_mon)
    y = str(t.tm_year)
    dirpath = "D:/Data/" + y + "/" + y + "-" + m + "-" + d
    print(dirpath)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
