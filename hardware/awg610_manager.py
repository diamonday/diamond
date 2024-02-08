import time
from threading import Thread
import numpy as np
import visa
from socket import SOL_SOCKET, SO_KEEPALIVE
from ftplib import FTP, error_temp
#from waveform import *

# UI
from traits.api import HasTraits, Array, Range, Float, Bool, Int, Long, Str, Enum, Button, Property, Instance, on_trait_change
from traitsui.api import View, Tabbed, VGroup, HGroup, Item, UItem, TextEditor, EnumEditor
from traitsui.ui_editors.array_view_editor import ArrayViewEditor


from copy import deepcopy

class AWG610( object ):
    """Controller for the Tektronix AWG610 device.
    
    SCPI commands are issued via gpib.
    See device manual for command documentation.
    File management is done via FTP.
    
    """
    
    def __init__(self, gpib='GPIB1::1::INSTR',
                        ftp='192.168.2.7',
                        socket=('192.168.2.7',4000) ): #unsure about this, since socket seems not used
        
        self.socket_addr = socket
        # set ftp parameters
        self.ftp_addr = ftp
        self.ftp_user = '\r'
        self.ftp_pw = '\r'
        self.ftp_cwd = '/main/waves'
        self.ftp_manager = FTPManager(self)
        self.todo = -1
        self.done = -1
        # setup gpib connection
        self.gpib_addr = gpib
        self.gpib = visa.instrument(self.gpib_addr)
        self.gpib.timeout = 5.0
        
    def __del__(self):
        self.gpib.close()
    
    # ____________
    # File Management
    
    def upload(self, files):
        # allow single files
        if not isinstance(files, (list, tuple)):
            files = [files]
    
        # opens up new ftp connections in separate threads
        self.todo = len(files)
        self.done = 0
        for file in files:
            self.ftp_manager.upload(file)
    
    def delete_all(self):
        """Remove all files from the AWG's CWD.
        """
        self.ftp_manager.delete_all()
        
    # ____________
    # Operation Commands
    
    def tell(self, command):
        """Send a command string to the AWG."""
        self.gpib.write(command)
        
    def ask(self, query):
        """Send a query string to AWG and return the response."""
        self.gpib.write(query)
        try:
            res = self.gpib.read()
        except visa.VisaIOError as e:
            res = ''
            if 'Timeout' in e.message:
                res = 'No response from AWG for: "' + query + '"'
            else:
                raise e
        return res
    
    def run(self):
        self.tell('AWGC:RUN')
    
    def stop(self):
        self.tell('AWGC:STOP')
        
    def force_trigger(self):
        self.tell('*TRG')
    
    def force_event(self, bitcode):
        self.tell('AWGC:EVEN:SOFT %i' %bitcode)
        
    def set_output(self, channel=1):
        """Set the output state of specified channels.
        
        channels - int with states 
                    0 for off, 1 for on
        
        """
        self.tell('OUTP1 %i' %channel ) # not sure on or 1
        
    def set_mode(self, mode):
        """Change the output mode.
        
        Options for mode (case-insensitive):
        continuous - 'C', 'CONT'
        triggered  - 'T', 'TRIG'
        gated      - 'G', 'GAT'
        sequence   - 'S', 'SEQ'
        
        """
        look_up = {'C' : 'CONT', 'CON' : 'CONT', 'CONT' : 'CONT',
                   'T' : 'TRIG', 'TRI' : 'TRIG', 'TRIG' : 'TRIG',
                   'G' : 'GAT' , 'GAT' : 'GAT' , 'GATE' : 'GAT' ,
                   'E' : 'ENH' , 'ENH' : 'ENH' , 'ENHA' : 'ENH' ,
                  }
        self.tell('AWGC:RMOD %s' % look_up[mode.upper()])
    
    def set_sampling(self, frequency):
        """ Set the output sampling rate.
        
        """
        frequency *= 1e-9
        self.tell('SOUR:FREQ %.4GGHz' % frequency)
    
    def set_vpp(self, voltage):
        """ Set output peak-to-peak voltage of specified channel(only 1 channel).
            
        """
        self.tell('SOUR1:VOLT %.4GV' % voltage)
    
    def load(self, filename, channel=1, cwd='\waves', block=False):
        """Load sequence or waveform file into RAM, preparing it for output.
        Waveforms and single channel sequences can be assigned to each or both
        channels. Double channel sequences must be assigned to channel 1.
        The AWG's file system is case-sensitive.
        
        """
        self.tell('SOUR%i:FUNC:USER "%s/%s"' % (channel, cwd, filename))
        
        # block thread until the operation is complete
        while block:
            try:
                self.ask('*OPC?')
                self.tell('SYST:BEEP')
                block = False
            except visa.VisaIOError as e:
                if not 'Timeout' in e[0]: raise e
    
    def managed_load(self, filename, channel=1, cwd='\waves'):
        print(cwd)
        self.ftp_manager.load(filename, channel, cwd)
        
    def get_func(self, channel=1):
        res = self.ask('SOUR%i:FUNC:USER?' %channel)
        # res ~ '"/\\waves/0_MAIN.SEQ","MAIN"'
        return res.split(',')[0].split('/')[-1][:-1] # return ~ '0_MAIN.SEQ'
        
    def reset(self):
        """ Reset the AWG settings. """
        self.tell('*RST')
        
class FTPThread(Thread):
    """ Thread, which opens a new FTP connection.
    """
    def __init__(self, awg):
        # emulate state stuff
        class Foo(): pass
        self.file = Foo()
        self.file.state = 'compiling'
        
        self.awg = awg
        super(FTPThread, self).__init__()
        self.daemon = True
        self.file.state = 'ready'
        
    def setup_ftp(self):
        self.ftp = FTP(self.awg.ftp_addr, timeout=2.0)
        self.ftp.set_pasv(False)
        self.ftp.login(self.awg.ftp_user, self.awg.ftp_pw)
        self.ftp.cwd(self.awg.ftp_cwd)
        
    def run(self):
        try:
            self.setup_ftp()
            #print('after setup_ftp')
            self.task()
            #print('after task')
            self.ftp.close()
            #print('after ftp.close')
            self.file.state = 'finished'
        except Exception as e:
            try:
                self.ftp.close()
            except AttributeError as a:
                pass
            self.file.state = 'error'
            # dont raise error_temp
            # if not isinstance(e, error_temp):
            raise e
            
    def task(self): pass

class UploadThread(FTPThread):
    
    def __init__(self, awg, file):
        super(UploadThread, self).__init__(awg)
        self.file = file
    
    def task(self):
        self.file.seek(0)
        self.ftp.storbinary('STOR ' + self.file.name, self.file)

class DeleteAllThread(FTPThread):
    
    def task(self):
        filelist = self.ftp.nlst()
        try:
            filelist.remove('.')
            filelist.remove('..')
        except ValueError:
            pass
        self.awg.tell('MMEM:CDIR "%s"' % self.awg.ftp_cwd)
        for file in filelist:
            self.awg.tell('MMEM:DEL "%s"' % file)
            #self.ftp.delete(file)
        time.sleep(0.5)
            
class FTPManager(Thread):
    """
    This Thread will prevent/workaround 421 session limit.
    It is only able to do to tasks, uploading files and deleting all files.
    """
    
    def __init__(self, awg):
        self.awg = awg
        self.threads = []
        self.clients = 0
        self.max_clients = 4
        self.awg.done = -1
        self.awg.todo = -1
        self.abort = False
        self.pause_set = False
        self.paused = False
        self.load_file = None
        super(FTPManager, self).__init__()
        self.daemon = True
        self.start()
        
    def upload(self, file):
        ut = UploadThread(self.awg, file)
        self.threads.append(ut)
        
    def delete_all(self):
        dt = DeleteAllThread(self.awg)
        self.threads.append(dt)
        
    def load(self, filename, channel=1, cwd='\waves'):
        self.load_file = {
            'filename': filename,
            'channel' : channel,
            'cwd'     : cwd,
        }
        
    def reset(self):
        self.pause_set = True
        self.threads = []     # really bad practice!!! TODO: make stappable threads - stop and join them
        
        while not self.paused:
            time.sleep(0.1)
        self.clients = 0
        self.awg.done = -1
        self.awg.todo = -1
        self.pause_set = False
        
    def stop(self):
        self.abort = True
        
    def run(self):
        # Event loop 
        while True:
            # check list of threads repeatedly
            for thr in self.threads:
                if self.abort: return
                # ignore running threads
                if not thr.is_alive():
                    
                    # Case DeleteAllThread:
                    if isinstance(thr, DeleteAllThread):
                        # start a DeleteAllThread
                        if thr.file.state == 'ready' and self.clients == 0:
                            thr.start()
                            self.clients += self.max_clients
                            #time.sleep(0.001)
                        # remove finished DeleteAllThread
                        elif thr.file.state == 'finished':
                            self.clients = 0
                            self.threads.remove(thr)
                        # restart failed DeleteAllThread
                        elif thr.file.state == 'error':
                            self.clients = 0
                            self.threads.remove(thr)
                            self.delete_all()
                            
                    # Case UploadThread:
                    elif isinstance(thr, UploadThread):
                        # start new UploadThread
                        if thr.file.state == 'ready' and self.clients < self.max_clients:
                            thr.start()
                            self.clients += 1
                            #time.sleep(0.001)
                        # remove finished UploadThread
                        elif thr.file.state == 'finished':
                            self.clients -= 1
                            self.threads.remove(thr)
                            self.awg.done += 1
                        # restart failed UploadThread
                        elif thr.file.state == 'error':
                            self.clients -= 1
                            thr.file.seek(0)
                            thr.file.state = 'ready'
                            self.upload(thr.file)
                            self.threads.remove(thr)
                # stop threads if abort is set
                time.sleep(0.001)
            # check if there is something to load into RAM
            
            if len(self.threads) == 0 and self.awg.done != -1 and self.load_file is not None:
                f = self.load_file
                self.awg.load(f['filename'], f['channel'], f['cwd'], block=True)
                self.load_file = None
            if self.pause_set:
                self.paused = True
                while self.pause_set:
                    time.sleep(0.1)
                self.paused = False
                

def getFile(s):
    s_split = s.split(',')
    fname = s_split[0]
    # s_split[1] is DIR or empty, hence s_split[2]
    if s_split[1] == 'DIR':
        ftype = 'Directory'
    elif s_split[1] == '':
        ftype = 'File'
    else:
        ftype = 'Unknown'
    fsize = s_split[2]

    if fname == '':
        fname = '-'
    if fsize == '':
        fsize = '-'

    return fname, ftype, fsize

# When there are many files, deleting or query can both be quite time-consuming
# It deserves a separate thread
class CatThread(Thread):

    def __init__(self,awg):
        self.awg = awg
        super(CatThread, self).__init__()

    def run(self):
        self.awg.cat_thread_state = 'Loading...'
        raw = self.awg.ask('MMEM:CAT?')
        self.awg.decode_flist(raw)
        self.awg.cat_empty = False
        self.awg.cat_thread_state = 'Idle'


class DeleteByRangeThread(Thread):

    def __init__(self,awg,head,tail):
        self.awg = awg
        self.head = head
        self.tail = tail
        super(DeleteByRangeThread, self).__init__()

    def run(self):
        self.awg.del_byRange_thread_state = 'Loading...'
        self.awg.delete_range(self.head,self.tail)
        self.awg.del_byRange_thread_state = 'Idle'



class AWGManager610( HasTraits, AWG610 ):
    
    todo         = Int()
    done         = Int()
    progress     = Property(trait=Float, depends_on=['todo', 'done'],  format_str='%.1f')
    abort_button = Button(label='abort upload')
    recent_upload = Str('', label='Wave', desc='last set waveform or sequence')
    
    # OUTP
    outp_table = {
        '1':True, 'On':True,
        '0':False, 'Off':False,
    }
    outp1  = Bool(False, label='CH 1', desc='enable CH1')
    outp_sbutton = Button(label='Set')
    outp_qbutton = Button(label='Query')

    # AWGC:RMOD
    rmod_table = {
        'CONT':'Continuous',
        'TRIG':'Triggered',
        'GATE':'Gated',
        'ENH':'Enhanced',
    }
    rmod = Enum(['ENH', 'CONT', 'TRIG', 'GATE'],
        label='Options',
        desc='the run mode of AWG',
        editor=EnumEditor(
            values=rmod_table,
            format_str='%s mode'
        )
    )
    rmod_sbutton = Button(label='Set')
    rmod_qbutton = Button(label='Query')
    rmod_query = Str('', label='Result')

    # AWGC:RUN
    run_button  = Button(label='Run', desc='Run')
    # AWGC:STOP
    stop_button = Button(label='Stop', desc='Stop')

    # *TRG or TRIG[:SEQ][:IMM]
    trig_button = Button(label='Force Trigger', desc='Force Trigger')

    # TRIG:LEV
    lev = Range(
        value=1.0, low=-5.0, high=5.0, label='Level [V]', desc='Trigger Level',
        mode='text', auto_set=False, enter_set=True
    )
    lev_sbutton = Button(label='Set')
    lev_qbutton = Button(label='Query')

    # TRIG:SLOP
    slop_table = {
        'POS':'Positive',
        'NEG':'Negative',
    }
    slop = Enum(['NEG','POS'],
        label='Set',
        desc='the trigger mode',
        editor=EnumEditor(
            values=slop_table,
            format_str='%s'
        )
    )
    slop_sbutton = Button(label='Set')
    slop_qbutton = Button(label='Query')
    #slop_query = Str('', label='Result')

    # MMEM:CDIR
    cdir = Str('', label='Directory Name')
    cdir_sbutton = Button(label='Set')
    cdir_qbutton = Button(label='Query')

    mdir = Str('', label='Directory Name')
    mdir_sbutton = Button(label='Make')

    del_byName = Str('', label='By Name')
    del_byName_sbutton = Button(label='Delete')

    del_byRange = Str(
        value='', label='By Range',
        desc='e.g. input 0:7 to delete the 0th - 7th files'
    )
    del_byRange_sbutton = Button(label='Delete')
    del_byRange_thread_state = Str('Idle')

    # MMEM:CAT
    #cat_list = List(Instance(AWGFile))
    cat_file_2dlist = Array()
    cat_capacity = Long(0, label='Total memory capacity')
    cat_available = Long(0, label='Available memory')
    cat_thread_state = Str('Idle')
    cat_qbutton = Button(label='Query')

    # SOUR

    # SOUR<x>:FUNC:USER
    func1 = Str('', label='CH1')
    func_qbutton = Button(label='Query')
    func_sbutton = Button(label='Set')

    # SOUR<x>:VOLT:LEV
    volt_lev1 = Range(
        value=2.0, low=0.02, high=2.0, label='CH1 [V]', desc='Voltage Level',
        mode='text', auto_set=False, enter_set=True
    )
    volt_lev_qbutton = Button(label='Query')
    volt_lev_sbutton = Button(label='Set')

    # SOUR<x>:VOLT:AMPL
    volt_ampl1 = Range(
        value=1.0, low=0.0, high=1.0, label='CH1', desc='Voltage Level',
        mode='text', auto_set=False, enter_set=True
    )
    volt_ampl_qbutton = Button(label='Query')
    volt_ampl_sbutton = Button(label='Set')


    def __init__(self,
        gpib='GPIB1::1::INSTR',
        ftp='192.168.2.7',
        socket=('192.168.2.7',4000),
        **kwargs
    ):
        AWG610.__init__(self, gpib, ftp, socket)
        HasTraits.__init__(self, **kwargs)
        self._outp_qbutton_fired()
        self.cat_empty = True

    ##### FTP #####

    def sync_upload_trait(self, client, trait_name):
        self.sync_trait('progress', client, trait_name, mutual=False)
    
    def _get_progress(self):
        return (100.0 * self.done) / self.todo
        
    def _abort_button_fired(self):
        self.ftp_manager.reset()

    ##### AWGC #####

    def _run_button_fired(self):
        self.run()
        
    def _stop_button_fired(self):
        self.stop()
    
    def _rmod_sbutton_fired(self):
        self.set_mode(self.rmod)

    def _rmod_qbutton_fired(self):
        result = self.ask('AWGC:RMOD?')
        self.rmod = result


    ##### TRIG #####

    def _trig_button_fired(self):
        self.force_trigger()
    
    def _lev_sbutton_fired(self):
        self.tell('TRIG:LEV %f' % self.lev)

    def _lev_qbutton_fired(self):
        self.lev = float(self.ask('TRIG:LEV?'))

    def _slop_sbutton_fired(self):
        self.set('TRIG:SLOP %s' % self.slop)

    def _slop_qbutton_fired(self):
        result = self.ask('TRIG:SLOP?')
        self.slop = result

    ##### OUTP #####

    def _outp_qbutton_fired(self):
        result1 = self.ask('OUTP1:STAT?')
        self.outp1 = self.outp_table[result1]
        
    def _outp_sbutton_fired(self):
        if self.outp1:
            self.tell('OUTP1 1')

    ##### SOUR #####

    def _func_qbutton_fired(self):
        result1 = self.ask('SOUR1:FUNC:USER?')
        s1 = result1.split('","')[0]
        self.func1 = s1.replace('"','').replace('\\','')

    def _func_sbutton_fired(self):
        self.tell('SOUR1:FUNC:USER % "%s"' % self.func1)

    def _volt_lev_qbutton_fired(self):
        result1 = self.ask('SOUR1:VOLT:LEV?')
        self.volt_lev1 = float(result1)

    def _volt_lev_sbutton_fired(self):
        self.tell('SOUR1:VOLT:LEV % ' % self.func1)
    
    def _volt_ampl_qbutton_fired(self):
        result1 = self.ask('SOUR1:VOLT:AMPL?')
        self.volt_ampl1 = float(result1)

    def _volt_ampl_sbutton_fired(self):
        self.tell('SOUR1:VOLT:AMPL % ' % self.volt_ampl1)

    ##### MMEM #####

    def load(self, filename, channel=1, cwd='\waves', block=False):
        self.recent_upload = filename
        super(AWGManager610, self).load(filename, channel, cwd, block)
    
    def _cdir_sbutton_fired(self):
        try:
            self.tell('MMEM:CDIR "%s"' % self.cdir)
        except:
            print('Failed to change directory to %s' % self.cdir)

    def _cdir_qbutton_fired(self):
        result = self.ask('MMEM:CDIR?')
        self.cdir = result.replace('"','')

    def _cat_qbutton_fired(self):
        cat_thread = CatThread(self)
        cat_thread.start()

    def _cat_file_2dlist_default(self):
        return [['-','-','-']]

    def decode_flist(self, raw_str):
        str_list = raw_str.split('","')
        fname_list = []
        ftype_list = []
        fsize_list = []

        # The first entry is expected to be:
        # [Total capacity],[available],"[Fname1],[DIR or empty],[Fsize1]
        
        # Split by " at the center
        s0 = str_list[0].split('"')

        s00_split = s0[0].split(',')
        capacity = int(s00_split[0])
        available = int(s00_split[1])

        s01 = s0[1]
        fn, ft, fs = getFile(s01)
        fname_list.append(fn)
        ftype_list.append(ft)
        fsize_list.append(fs)

        # This for-loop only run if there're >= 3 files 
        for s_raw in str_list[1:-1]:
            fn, ft, fs = getFile(s_raw)
            fname_list.append(fn)
            ftype_list.append(ft)
            fsize_list.append(fs)
        
        if len(str_list) > 1:
            s_las = str_list[-1].replace('"','')
            fn, ft, fs = getFile(s_las)
            fname_list.append(fn)
            ftype_list.append(ft)
            fsize_list.append(fs)

        self.cat_capacity = capacity
        self.cat_available = available
        self.cat_file_2dlist = np.array([fname_list,ftype_list,fsize_list]).T

    def _mdir_sbutton_fired(self):
        self.tell('MMEM:MDIR "%s"' % self.mdir)
    
    def _del_byName_sbutton_fired(self):
        self.tell('MMEM:DEL "%s"' % self.del_byName)

    def _del_byRange_sbutton_fired(self):
        proceed = True # Decide whether to delete files

        if self.cat_empty is False:
            head,tail = self.del_byRange.split(':')
            if head.replace(' ','') == '':
                head = 0
            else:
                try:
                    head = int(head)
                except:
                    proceed = False
                    print(head, 'is not an integer')
            if tail.replace(' ','') == '':
                tail = len(self.cat_file_2dlist)
            else:
                try:
                    tail = int(tail)
                except:
                    proceed = False
                    print(tail, 'is not an integer')
        else:
            proceed = False
            print('Please read the list of files for once')
        
        if proceed:
            del_thread = DeleteByRangeThread(self,head,tail)
            del_thread.start()


    def delete_range(self,head,tail):  
        for row in self.cat_file_2dlist[head:tail]:
            fname = row[0]
            self.tell('MMEM:DEL "%s"' % fname)


    cat_array_editor = ArrayViewEditor(
        titles = ['File name', 'Type', 'File size (Bytes)'],
        show_index = True,
    )

    view = View(
        Tabbed(
            VGroup(
                HGroup(
                    Item('progress', width=40, style='readonly', format_str='%.1f'),
                    Item('todo', width=40, style='readonly'),
                    UItem('abort_button'),
                    Item('recent_upload', style='readonly'),
                    label='FTP Information',
                    show_border=True,
                ),
                HGroup(
                    Item('outp1'),
                    UItem('outp_sbutton'),
                    UItem('outp_qbutton'),
                    label='Output (OUTP)',
                    show_border=True,
                ),
                HGroup(
                    HGroup(
                        UItem('run_button'),
                        UItem('stop_button'),
                    ),
                    HGroup(
                        Item('rmod', width=-80),
                        UItem('rmod_sbutton'),
                        UItem('rmod_qbutton'),
                        label='Run mode (RMOD)',
                        show_border=True,
                    ),
                    label='AWG Control (AWGC)',
                    show_border=True,
                ),
                VGroup(
                    HGroup(
                        Item('trig_button', show_label=False),
                        HGroup(
                            Item('lev', width=-40),
                            UItem('lev_sbutton'),
                            UItem('lev_qbutton'),
                            label='Level (LEV)',
                            show_border=True,
                        ),                        
                    ),
                    HGroup(
                        Item('slop', width=-80),
                        UItem('slop_sbutton'),
                        UItem('slop_qbutton'),
                        label='Slope (SLOP)',
                        show_border=True,
                    ),
                    label='Trigger (TRIG)',
                    show_border=True,
                ),
                VGroup(
                    HGroup(
                        Item('func1', width=-140),
                        UItem('func_sbutton'),
                        UItem('func_qbutton'),
                        label='Loaded sequence (FUNC:USER)',
                        show_border=True,
                    ),
                    HGroup(
                        Item('volt_lev1', width=-40),
                        UItem('volt_lev_sbutton'),
                        UItem('volt_lev_qbutton'),
                        label='Voltage level (VOLT:LEV)',
                        show_border=True,
                    ),
                    HGroup(
                        Item('volt_ampl1', width=-40),
                        UItem('volt_ampl_sbutton'),
                        UItem('volt_ampl_qbutton'),
                        label='Amplitude (VOLT:AMPL)',
                        show_border=True,
                    ),
                    label='Source (SOUR)',
                    show_border=True,
                ),
                label='Control'
            ),
            VGroup(
                HGroup(
                    Item('cdir', width=-200),
                    UItem('cdir_sbutton'),
                    UItem('cdir_qbutton'),
                    label='Current Directory (CDIR)',
                    show_border=True,
                ),
                HGroup(
                    Item('mdir', width=-200),
                    UItem('mdir_sbutton'),
                    label='Make Directory (MDIR)',
                    show_border=True,
                ),
                VGroup(
                    HGroup(
                        Item('del_byName', width=-200),
                        UItem('del_byName_sbutton'),
                    ),
                    HGroup(
                        Item('del_byRange', width=-200),
                        UItem('del_byRange_sbutton'),
                        UItem('del_byRange_thread_state', style='readonly'),
                    ),
                    label='Delete File/Directory (DEL)',
                    show_border=True,
                ),
                VGroup(
                    VGroup(
                        HGroup(
                            UItem('cat_qbutton'),
                            UItem('cat_thread_state', style='readonly'),
                        ),
                        HGroup(
                            Item('cat_capacity', style='readonly', width=80),
                            Item('cat_available', style='readonly', width=80),
                        ),
                    ),
                    UItem('cat_file_2dlist', editor=cat_array_editor),
                    label='Catalog (CAT)',
                    show_border=True,
                ),
                label='Mass Memory (MMEM)',
                show_border=True,
            ),           
        ),
        title='AWG610 Manager',
        width=600,height=800,
        buttons=[], resizable=True
    )
# _____________________________________________________________________________
# EXCEPTIONS:
    # TODO

# _____________________________________________________________________________
# DEBUG SCRIPT:

if __name__ == '__main__':
    pass
