import numpy as np
import time
import logging
import os.path

import traceback

import hardware.api as ha
from hardware.waveformX import SequenceX, Sin, Idle

from traits.api import Instance, Property, Range, Float, Int, String, Bool, Array, List, Enum, Trait, Button, on_trait_change, cached_property, Code

from traitsui.api import View, Item, Group, HGroup, VGroup, Tabbed, VSplit, TextEditor, EnumEditor, Include#, RangeEditor, 
from traitsui.menu import Action, Menu, MenuBar

from enable.api import Component, ComponentEditor

from tools.emod import ManagedJob
from tools.utility import GetSetItemsMixin, GetSetItemsHandler

from traits.api import HasTraits

try:
	from tools.file_utility import save_file
except:
	from traitsui.file_dialog import save_file

from analysis.fitting import find_edge

from .common import *
from .ui import LinePlot, Histogram

from .temporary import VERSION, DATE

class QuantumNonDemolitionHandler(GetSetItemsHandler):

	def save_all(self, info):
		filename = save_file(title='Save All')

		if not filename:
			return

		info.object.save(filename + '.pys')

		try:
			info.object.save(filename + '.pyz')
		except:
			pass

		info.object.save_histogram(filename + '_histogram.png')

	def save_histogram(self, info):
		filename = save_file(title='Save Histogram')
		if not filename:
			return
		else:
			if filename.find('.png') == -1:
				filename = filename + '.png'
			info.object.save_histogram(filename)

class QuantumNonDemolition(ManagedJob, GetSetItemsMixin):
	# Measurement Name
	measurement_name = "QuantumNonDemolition"

	# Enable Check Statement
	idle_when = 'state in ("idle", "done", "error")'

	get_set_items = ['__doc__', 'measurement_name']

	# Temporary
	VERSION = VERSION
	DATE = DATE

	get_set_items.extend(['VERSION', 'DATE'])

	# Hardware Definitions
	time_tagger = ha.TimeTagger
	pg = ha.PulseGenerator()
	awg = ha.AWG70k()

	pg_ch_triga = "awga"
	pg_ch_trigb = "awg"
	pg_ch_mw = "mw_x"
	pg_ch_laser = "green"
	pg_ch_tt_laser = "laser"
	pg_ch_tt_seq = "sequence"

	tt_ch_apd = 0
	tt_ch_laser = 2
	tt_ch_seq = 3

	# Channels used in the measurement, can be 01, 10, 11. The variable is used in sequence loading.
	awg_channel = 01

	# AWG Detailed Parameters
	awg_pulse_length = Range(low=0.0, high=10000.0, value=21.0, desc="Trigger Pulse Length for triggering AWG", label="Trigger Pulse Length [ns]", mode='text', auto_set=False, enter_set=True)
	awg_trigger_delay = Range(low=0.0, high=1e6, value=1500.0, desc="Delay of AWG from Trigger to output", label="Trigger Delay [ns]", mode='text', auto_set=False, enter_set=True)
	awg_ch1_vpp = Range(low=0.0, high=0.5, value=0.5, desc="AWG Ch1 Vpp", label="AWG Ch1 Vpp [V]", mode='text', auto_set=False, enter_set=True)
	awg_ch2_vpp = Range(low=0.0, high=0.5, value=0.5, desc="AWG Ch2 Vpp", label="AWG Ch2 Vpp [V]", mode='text', auto_set=False, enter_set=True)
	awg_sampling_rate = Range(low=1.5e3, high=25.0e9, value=25.0e9, desc="AWG Sampling Rate", label="AWG Sampling Rate [S/s]", mode='text', auto_set=False, enter_set=True)

	awg_extra_view = HGroup(
		Item('awg_ch1_vpp', width=-50, enabled_when=idle_when),
		Item('awg_ch2_vpp', width=-50, enabled_when=idle_when),
		Item('awg_sampling_rate', width=-80, enabled_when=idle_when),
		Item('awg_pulse_length', width=-50, enabled_when=idle_when),
		Item('awg_trigger_delay', width=-50, enabled_when=idle_when),
		label='AWG Settings',
		show_border=True,
	)

	get_set_items.extend(['awg_pulse_length', 'awg_trigger_delay', 'awg_ch1_vpp', 'awg_ch2_vpp', 'awg_sampling_rate'])

	# Job Control
	resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')
	run_time  = Float(value=0.0, label='run time [s]',format_str='%.f')
	stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)

	job_view = HGroup(
		Item('submit_button', show_label=False, enabled_when=idle_when),
		Item('remove_button', show_label=False),
		Item('resubmit_button', show_label=False, enabled_when=idle_when),
		Item('priority'),
		Item('state', style='readonly'),
		Item('run_time', style='readonly', format_str='%i'),
		Item('stop_time'),
	)

	get_set_items.extend(['run_time', 'stop_time'])

	# AWG Sequence Loading
	reload_awg = Bool(True, label='reload', desc='Compile waveforms upon start up.')
	wfm_button = Button(label='Load', desc='Compile waveforms and upload them to the AWG.')
	upload_progress = Float(label='Upload progress', desc='Progress uploading waveforms', mode='text')

	awg_view = HGroup(
		Item('reload_awg', enabled_when=idle_when),
		Item('wfm_button', show_label=False, enabled_when=idle_when),
		Item('upload_progress', style='readonly', format_str='%i'),
		label='AWG',
		show_border=True,
	)

	# Readout
	mw_freq = Range(low=1e9, high=10e9, value=2.87e9, desc="MW Frequency", label="MW Freq [Hz]", mode='text', auto_set=False, enter_set=True)
	mw_amp = Range(low=0.0, high=0.5, value=0.1, desc="MW Amplitude [V]", label="MW Amplitude [V]", mode='text', auto_set=False, enter_set=True)
	mw_t_pi = Range(low=0.0, high=10000.0, value=100.0, desc="MW Pi", label="MW pi [ns]", mode='text', auto_set=False, enter_set=True)

	read_time = Range(low=0.0001, high=1, value=0.005, desc="Read Time [s]", label="Read Time [s]", mode='text', auto_set=False, enter_set=True)
	read_runs_round = Range(low=1, high=100000, value=1000, desc="Read Runs Per Round", label="Runs Per Round", mode='text', auto_set=False, enter_set=True)

	read_count_low = Range(low=0, value=200, high=1000, desc="Min Count", label="Min Count", mode='text', auto_set=False, enter_set=True)
	read_count_high = Range(low=200, value=400, high=2000, desc="Max Count", label="Max Count", mode='text', auto_set=False, enter_set=True)
	read_count_delta = Range(low=1, value=5, hight=100, desc="Delta Count", label="Delta Count", mode='text', auto_set=False, enter_set=True)

	save_time_trace = Bool(False, desc="Save Time Trace", label="Save Time Trace")
	time_trace_path = String("", desc="Time Trace Folder", label="Time Trace Folder")

	read_delay = Range(low=0.0, high=100.0, value=1.0, desc="Read Delay [s]", label="Read Delay [s]", mode='text', auto_set=False, enter_set=True)

	laser_length = Range(low=0.0,  high=10000.0, value=300.0, desc='Laser Length', label='Laser Length [ns]', mode='text', auto_set=False, enter_set=True)
	wait_length = Range(low=0.0,  high=10000.0, value=0.0, desc='Wait Length', label='Wait Length [ns]', mode='text', auto_set=False, enter_set=True)


	readout_view = Group(
		HGroup(
			Item("mw_freq", enabled_when=idle_when),
			Item("mw_amp", enabled_when=idle_when),
			Item("mw_t_pi", enabled_when=idle_when),
		),
		HGroup(
			Item("read_time", enabled_when=idle_when),
			Item("read_runs_round", enabled_when=idle_when),
			Item("read_count_low", enabled_when=idle_when),
			Item("read_count_high", enabled_when=idle_when),
			Item("read_count_delta", enabled_when=idle_when),
			Item("save_time_trace", enabled_when=idle_when),
			Item("time_trace_path", enabled_when=idle_when),
		),
		HGroup(
			Item("read_delay", enabled_when=idle_when),
			Item("laser_length", enabled_when=idle_when),
			Item("wait_length", enabled_when=idle_when),
		),
		label="Measurement and Readout",
		show_border=True
	)

	get_set_items.extend([
		'mw_freq', 'mw_amp', 'mw_t_pi',
		'read_time', 'read_runs_round', 'read_delay', 'save_time_trace', 'time_trace_path',
		'laser_length', 'wait_length',
		'read_count_low', 'read_count_high', 'read_count_delta'
	])

	# Visualization
	histogram = Instance(Histogram, factory=Histogram, kw={"dual": True})

	visualization_view = Tabbed(
		Group(
			Item("histogram", style="custom", show_label=False),
			label="Histogram",
			show_border=True,
		),
	)

	# Dummy Values
	runs = 0
	data_bins = Array()
	count_data = Array()
	sequence = []
	sequence_time = 0
	keep_data = False

	get_set_items.extend(['runs', 'data_bins', 'count_data', 'sequence', 'sequence_time', 'keep_data'])

	menu_bar = MenuBar(
		Menu(
			Action(action='save_all', name='Save All'),
			Action(action='save', name='Save (.pyd or .pys)'),
			Action(action='load', name='Load (.pyd or .pys)'),
			Action(action='save_histogram', name='Save Histogram (.png)'),
			Action(action='_on_close', name='Quit'),
			name='File'
		)
	)

	traits_view = View(
		Group(
			Include("job_view"),
			VSplit(
				Tabbed(
					VGroup(
						Include("awg_view"),
						Include("readout_view"),
						label="Main",
					),
					Group(
						Include("awg_extra_view"),
						label="Settings",
					),
				),
				Include("visualization_view"),
			),
		),
		menubar=menu_bar,
		handler=QuantumNonDemolitionHandler,
		title="Quantum Non Demolition Measurement",
		buttons=[],
		resizable=True,
		#width =-900,
		#height=-800,
	)

	def __init__(self):
		super(QuantumNonDemolition, self).__init__()

		self.awg.sync_upload_trait(self, 'upload_progress')

	# Job Control Functions
	def submit(self):
		"""Submit the job to the JobManager."""
		self.keep_data = False
		ManagedJob.submit(self)

	def resubmit(self):
		"""Submit the job to the JobManager."""
		self.keep_data = True
		ManagedJob.submit(self)

	def _resubmit_button_fired(self):
		"""React to start button. Submit the Job."""
		self.resubmit()

	# Plot Saving Functions
	def save_histogram(self, filename):
		self.save_figure(self.histogram.plot, filename)

	# Plot Update Functions
	@on_trait_change("data_bins")
	def update_histogram(self):
		if len(self.data_bins) == 0:
			return

		self.histogram.update_data("x", self.data_bins)

		if self.runs and len(self.count_data):
			self.histogram.update_data("y1", self.count_data[0])
			self.histogram.update_data("y2", self.count_data[1])
		
	@on_trait_change("count_data")
	def update_count_data_change(self):
		if len(self.count_data) == 0:
			return

		self.histogram.update_data("y1", self.count_data[0])
		self.histogram.update_data("y2", self.count_data[1])

	def apply_parameters(self):
		"""Apply the current parameters and decide whether to keep previous data."""

		data_bins = np.arange(self.read_count_low, self.read_count_high + self.read_count_delta, self.read_count_delta)
		count_data = np.zeros((2, data_bins.size + 2))

		if self.keep_data and np.all(data_bins == self.data_bins): # if the count binning is the same as previous, keep existing data
			pass
		else:
			self.run_time = 0.0
			self.runs = 0

			self.data_bins = data_bins
			self.count_data = count_data

		self.sequence = self.generate_sequence()

		# prepare awg
		if not self.keep_data:
			self.prepare_awg()

		self.keep_data = True

	def prepare_awg(self):
		if self.reload_awg:
			self.load_wfm()
		self.awg.set_vpp(self.awg_ch1_vpp, 0b01)
		self.awg.set_vpp(self.awg_ch2_vpp, 0b10)
		self.awg.set_sampling(self.awg_sampling_rate)

		self.awg.set_run_mode(self.awg.mode_ch1, channel=0b01)
		self.awg.set_run_mode(self.awg.mode_ch2, channel=0b10)

		self.awg.set_trigger(
			TRIG_source=[self.awg.trigger_ch1, self.awg.trigger_ch2], 
			TRIG_edge=[self.awg.trigger_edge_ATR, self.awg.trigger_edge_BTR], 
			TRIG_level=[self.awg.trigger_level_ATR, self.awg.trigger_level_ATR] # Copied from pulsed_awg1. Bug?
		)

	def _wfm_button_fired(self):
		self.reload_awg = True
		self.prepare_awg()

	def load_wfm(self):
		self.awg_waves = []
		self.awg_main_wave = ''

		self.generate_waveform()

		self.awg.upload(self.awg_waves)
		self.awg.makeseqx(self.awg_main_wave, self.awg_waves)

		if self.awg_channel == 01:
			self.awg.managed_load(self.awg_main_wave, 1)
		elif self.awg_channel == 10:
			self.awg.managed_load(self.awg_main_wave, 2)
		elif self.awg_channel == 11:
			self.awg.managed_load(self.awg_main_wave, 11)
		else:
			pass

		self.reload_awg = False

	def generate_sequence(self):
		_sequence_mw = [
			((self.pg_ch_trigb,), self.awg_pulse_length),
			(tuple(), self.awg_trigger_delay + self.mw_t_pi),
			((self.pg_ch_laser,), self.laser_length),
			(tuple(), self.wait_length)
		] * 200

		_sequence_nomw = [
			(tuple(), self.awg_pulse_length),
			(tuple(), self.awg_trigger_delay + self.mw_t_pi),
			((self.pg_ch_laser,), self.laser_length),
			(tuple(), self.wait_length)
		] * 200

		return _sequence_mw, _sequence_nomw

	def generate_waveform(self):
		name = self.measurement_name

		self.awg_sequence = SequenceX(name)

		sampling = self.awg_sampling_rate

		_mw_freq = self.mw_freq / sampling
		_t_pi = self.mw_t_pi * (sampling / 1e9)

		R_x = Sin(_t_pi, freq=_mw_freq, amp=self.mw_amp, phase=0.0)

		mw_wavename = "READ1_1"

		mw_sequence = [R_x]

		if _t_pi < 2400:
			_idle = 2400 - _t_pi
			_fill_sequence = Idle(_idle)
			
			mw_sequence.append(_fill_sequence)

		self.awg_sequence.addWaveform(mw_wavename, mw_sequence)

		self.awg_sequence.seqBox[self.measurement_name].writeStep(
			(mw_wavename, "Waveform"),
			WaitInput="TrigB",
			Repeat="Once",
			EventJumpInput="None",
			GoTo=1
		)

		self.awg_waves.append(self.awg_sequence.wavBox[mw_wavename])

		self.awg_sequence.seqBox[name].compileSML() 
		self.awg_waves.append(self.awg_sequence.seqBox[name])
		self.awg_main_wave = self.awg_sequence.name

	def start_up(self):
		self.pg.Night()

		self.awg.set_output(self.awg.channel)
		self.awg.run()
		time.sleep(1)
		self.awg.update_run_state()

	def shut_down(self):		
		self.pg.Light()

		self.awg.stop()
		self.awg.set_output(0b00)
		time.sleep(1)
		self.awg.update_run_state()

	def _run(self):

		_exit_state = ['error', 'idle', 'done', 'error']
		_exit_reason = 0

		try:
			self.state = 'run'
			self.apply_parameters()

			self.start_up()

			_counter = self.time_tagger.Counter(self.tt_ch_apd, int(self.read_time * 1e12), self.read_runs_round)

			while self.run_time < self.stop_time:
				_start_time = time.time()

				self.pg.Sequence(self.sequence[0], loop=True)
				time.sleep(self.read_delay)
				
				time.sleep(self.read_time * self.read_runs_round * 1.1)
				
				_data = _counter.getData()

				if self.save_time_trace:
					_path = os.path.normpath(self.time_trace_path) + "/" + "MW_%i.npy" % self.runs
					
					np.save(_path, _data)

				for i in range(0, self.read_runs_round):
					_sum = _data[i]

					if _sum < self.read_count_low:
						self.count_data[0][0] += 1
					elif _sum > self.read_count_high:
						self.count_data[0][-1] += 1
					else:
						_ind = (_sum - self.read_count_low) // self.read_count_delta
						
						self.count_data[0][_ind + 1] += 1

				self.pg.Sequence(self.sequence[1], loop=True)
				time.sleep(self.read_delay)
				
				time.sleep(self.read_time * self.read_runs_round * 1.1)
				
				_data = _counter.getData()

				if self.save_time_trace:
					_path = os.path.normpath(self.time_trace_path) + "/" + "NoMW_%i.npy" % self.runs
					
					np.save(_path, _data)

				for i in range(0, self.read_runs_round):
					_sum = _data[i]

					if _sum < self.read_count_low:
						self.count_data[1][0] += 1
					elif _sum > self.read_count_high:
						self.count_data[1][-1] += 1
					else:
						_ind = (_sum - self.read_count_low) // self.read_count_delta
						
						self.count_data[1][_ind + 1] += 1

				self.runs += 1

				self.trait_property_changed('count_data', self.count_data)

				self.run_time += time.time() - _start_time

				if self.thread.stop_request.isSet():
					_exit_reason = 1
					break

			else:
				_exit_reason = 2

		except Exception as e:
			traceback.print_exc()
			_exit_reason = 3

		finally:
			self.shut_down()
			del _counter

		self.state = _exit_state[_exit_reason]