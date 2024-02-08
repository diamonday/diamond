import numpy as np

from traits.api import Range, Int, Float, Bool, Array, Instance, Enum, on_trait_change, Button
from traitsui.api import View, Item, Tabbed, HGroup, VGroup, VSplit, EnumEditor, TextEditor

import logging
import time

from hardware.api import PulseGenerator, TimeTagger, Microwave, MicrowaveB

from tools.emod import ManagedJob

from pulsed import Pulsed

import random

from tools.utility import GetSetItemsMixin

import traceback

class DEER3pi2(Pulsed):

	measurement_type = 'deer_3pi2'
	
	mw1_power = Range(low= -100., high=25., value=5.0, desc='MW1 Power [dBm]', label='MW1 Power [dBm]', mode='text', auto_set=False, enter_set=True)
	mw1_frequency = Range(low=1., high=20.e9, value=2.61e9, desc='MW1 Frequency [Hz]', label='MW1 Frequency [Hz]', mode='text', auto_set=False, enter_set=True)
	t_mw1_pi2 = Range(low=1., high=100000., value=83., desc='length of pi/2 pulse of mw1 [ns]', label='mw1 pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
	t_mw1_pi = Range(low=1., high=100000., value=166., desc='length of pi pulse of mw1 [ns]', label='mw1 pi [ns]', mode='text', auto_set=False, enter_set=True)
	t_mw1_3pi2 = Range(low=1., high=100000., value=250., desc='length of 3pi/2 pulse of mw1 [ns]', label='mw1 3pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
	mw2_power = Range(low= -100., high=25., value=7.0, desc='MW2 Power [dBm]', label='MW2 Power [dBm]', mode='text', auto_set=False, enter_set=True)
	rf_begin = Range(low=1, high=20e9, value=100.0e6, desc='Start Frequency [Hz]', label='Begin [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
	rf_end = Range(low=1, high=20e9, value=400.0e6, desc='Stop Frequency [Hz]', label='End [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
	rf_delta = Range(low=1e-3, high=20e9, value=2.0e6, desc='frequency step [Hz]', label='Delta [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
	t_mw2_pi = Range(low=1., high=100000., value=90., desc='length of pi pulse of mw2 [ns]', label='mw2 pi [ns]', mode='text', auto_set=False, enter_set=True)
	laser = Range(low=1., high=10000., value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
	wait = Range(low=1., high=10000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
	tau = Range(low=1., high=100000., value=300., desc='tau [ns]', label='tau [ns]', mode='text', auto_set=False, enter_set=True)
	seconds_per_point = Range(low=20e-7, high=1, value=0.2, desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)

	references = Bool(True, label='References')
	references_tau_mix = Range(low=0, high=1e7, label='REF0 Tau [ns]', mode='text', auto_set=False, enter_set=True)

	sweeps_per_point = Int()
	run_time = Float(value=0.0)

	frequencies = Array(value=np.array((0., 1.)))   
	
	randomize = Bool(True)

	cap_runs = Bool(True)
	cap_runs_margin = Range(low=1, high=99, value=5, label="Cap Runs Margin [%]", mode='text', auto_set=False, enter_set=True)

	runs_number = np.array((0,))

	def generate_sequence(self, type=None):
		_common_mw_t = min(self.t_mw1_pi, self.t_mw2_pi)
		_sec_mw_t2 = abs(self.t_mw1_pi - self.t_mw2_pi) / 2

		if self.t_mw1_pi > self.t_mw2_pi:
			_sec_mw = 'mw_a'
		else:
			_sec_mw = 'mw_b'

		if type == 'pi2':
			_sequence = 100 * [ (['mw_a'], self.t_mw1_pi2), ([], self.tau), ([_sec_mw], _sec_mw_t2), (['mw_a', 'mw_b'], _common_mw_t), ([_sec_mw], _sec_mw_t2), ([], self.tau), (['mw_a'], self.t_mw1_pi2), (['laser', 'aom'], self.laser), ([], self.wait), (['sequence'], 100) ]
		elif type == '3pi2':
			_sequence = 100 * [ (['mw_a'], self.t_mw1_pi2), ([], self.tau), ([_sec_mw], _sec_mw_t2), (['mw_a', 'mw_b'], _common_mw_t), ([_sec_mw], _sec_mw_t2), ([], self.tau), (['mw_a'], self.t_mw1_3pi2), (['laser', 'aom'], self.laser), ([], self.wait), (['sequence'], 100) ]
		elif type == 'pi2_ref_hahn':
			_sequence = 100 * [ (['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a'], self.t_mw1_pi), ([], self.tau), (['mw_a'], self.t_mw1_pi2), (['laser', 'aom'], self.laser), ([], self.wait), (['sequence'], 100) ]
		elif type == '3pi2_ref_hahn':
			_sequence = 100 * [ (['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a'], self.t_mw1_pi), ([], self.tau), (['mw_a'], self.t_mw1_3pi2), (['laser', 'aom'], self.laser), ([], self.wait), (['sequence'], 100) ]
		elif type == 'pi2_ref_mix':
			_sequence = 100 * [ (['mw_a'], self.t_mw1_pi2), ([], self.references_tau_mix), (['mw_a'], self.t_mw1_pi), ([], self.references_tau_mix), (['mw_a'], self.t_mw1_pi2), (['laser', 'aom'], self.laser), ([], self.wait), (['sequence'], 100) ]
		elif type == '3pi2_ref_mix':
			_sequence = 100 * [ (['mw_a'], self.t_mw1_pi2), ([], self.references_tau_mix), (['mw_a'], self.t_mw1_pi), ([], self.references_tau_mix), (['mw_a'], self.t_mw1_3pi2), (['laser', 'aom'], self.laser), ([], self.wait), (['sequence'], 100) ]
		elif type == 'refs':
			_sequence = 100 * [\
				(['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a'], self.t_mw1_pi), ([], self.tau), (['mw_a'], self.t_mw1_pi2), (['laser', 'aom'], self.laser), ([], self.wait),
				(['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a'], self.t_mw1_pi), ([], self.tau), (['mw_a'], self.t_mw1_3pi2), (['laser', 'aom'], self.laser), ([], self.wait),
				(['mw_a'], self.t_mw1_pi2), ([], self.references_tau_mix), (['mw_a'], self.t_mw1_pi), ([], self.references_tau_mix), (['mw_a'], self.t_mw1_pi2), (['laser', 'aom'], self.laser), ([], self.wait),
				(['mw_a'], self.t_mw1_pi2), ([], self.references_tau_mix), (['mw_a'], self.t_mw1_pi), ([], self.references_tau_mix), (['mw_a'], self.t_mw1_3pi2), (['laser', 'aom'], self.laser), ([], self.wait),
				(['sequence'], 100)
			]
		else:
			_sequence = 100 * [ (['laser', 'aom'], self.laser), ([], self.wait), (['mw_a'], self.t_mw1_pi2), ([], self.tau), ([_sec_mw], _sec_mw_t2), (['mw_a', 'mw_b'], _common_mw_t), ([_sec_mw], _sec_mw_t2), ([], self.tau), (['mw_a'], self.t_mw1_pi2), (['laser', 'aom'], self.laser), ([], self.wait), (['mw_a'], self.t_mw1_pi2), ([], self.tau), ([_sec_mw], _sec_mw_t2), (['mw_a', 'mw_b'], _common_mw_t), ([_sec_mw], _sec_mw_t2), ([], self.tau), (['mw_a'], self.t_mw1_3pi2), (['sequence'], 100) ]

		return _sequence

	def apply_parameters(self):
		"""Apply the current parameters and decide whether to keep previous data."""

		frequencies = np.arange(self.rf_begin, self.rf_end + self.rf_delta, self.rf_delta)
		n_bins = int(self.record_length / self.bin_width)
		time_bins = self.bin_width * np.arange(n_bins)
		sequence = self.generate_sequence()

		if not (self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins) and np.all(frequencies == self.frequencies)): # if the sequence and time_bins are the same as previous, keep existing data
			self.count_data = np.zeros((2 * len(frequencies) + 4 * self.references, n_bins))
			self.run_time = 0.0
			self.runs_number = np.zeros(2 * len(frequencies) + 4 * self.references)

		self.frequencies = frequencies
		self.sequence = sequence 
		self.time_bins = time_bins
		self.n_bins = n_bins

		_max_flip_t = max(self.t_mw1_pi, self.t_mw2_pi)

		if self.randomize:
			self.sweeps_per_point = int(np.max((1, int(self.seconds_per_point * 1e9 / (self.laser + self.wait + 2 * (self.t_mw1_3pi2 + self.tau) + _max_flip_t + 100)))))
		else:
			self.sweeps_per_point = int(np.max((1, int(self.seconds_per_point * 1e9 / (2 * (self.laser + self.wait + _max_flip_t) + 4 * (self.t_mw1_3pi2 + self.tau) + 100)))))

		self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.

	def _run(self):
		"""Acquire data."""

		# try to run the acquisition from start_up to shut_down
		try: 
			self.state = 'run'
			self.apply_parameters()

			PulseGenerator().Night()
			Microwave().setOutput(self.mw1_power, self.mw1_frequency)
			MicrowaveB().setPower(self.mw2_power)

			_len = self.frequencies.size

			_max_runs = int(self.sweeps_per_point * (1.0 - self.cap_runs_margin / 100.0))

			if self.randomize:
				_type_mapping = ['pi2', '3pi2']
				_itype_mapping = {'pi2':0, '3pi2': 1}

				tagger = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), 1, 0, 2, 3)
			else:
				PulseGenerator().Sequence(self.sequence)
				tagger = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), 2, 0, 2, 3)

			if self.cap_runs:
				tagger.setMaxCounts(_max_runs)

			while self.run_time < self.stop_time:

				if self.thread.stop_request.isSet():
					break

				if self.randomize:
					random.shuffle(_type_mapping)

				t_start = time.time()

				for j in range(1 + self.randomize):
					if self.randomize:
						_type = _type_mapping[j]
						_sequence = self.generate_sequence(_type)
						_rand_mapping = range(0, _len)
						random.shuffle(_rand_mapping)

						logging.getLogger().debug('Random Type Mapping: ' + str(_type_mapping) + '\nRandom Mapping: ' + str(_rand_mapping))

						PulseGenerator().Sequence(_sequence)

					for i in range(_len):
						if self.randomize:
							_freq = self.frequencies[_rand_mapping[i]]

						else:
							_freq = self.frequencies[i]

						logging.getLogger().debug('RF ' + 'Power: ' + str(self.mw2_power) + ', Freq: ' + str(_freq))

						MicrowaveB().setOutput(self.mw2_power, _freq)

						time.sleep(0.001)

						tagger.clear()

						time.sleep(self.seconds_per_point)

						
						_runs = tagger.getCounts()

						_iter = 0

						while _runs < _max_runs:
							_iter += 1
							time.sleep(self.seconds_per_point / 20)
							
							_runs = tagger.getCounts()
						
							if _iter >= 20:
								logging.getLogger().warn('Timed out. Insufficient measurement runs collected, RF Frequency: ' +  str(_freq))
								break

						_data = tagger.getData()

						if self.randomize:
							_pos = _rand_mapping[i] + _itype_mapping[_type] * _len
							logging.getLogger().debug('Data Insertion Position: ' + str(_pos))
							self.count_data[_pos] += _data[0]

							self.runs_number[_pos] += _runs

						else:
							self.count_data[i] += _data[0]
							self.count_data[i + _len] += _data[1]

							self.runs_number[i] += _runs
							self.runs_number[i + _len] += _runs

				if self.references:
					_wait_time = _max_runs * (4 * (self.laser + self.wait + self.t_mw1_pi + self.tau + self.references_tau_mix) + 6 * self.t_mw1_pi2 + 2 * self.t_mw1_3pi2 + 100) / 1e9

					_sequence = self.generate_sequence('refs')

					PulseGenerator().Night()
					del tagger
					
					MicrowaveB().setOutput(None, self.rf_begin)

					time.sleep(0.001)

					tagger = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), 4, 0, 2, 3)
					
					if self.cap_runs:
						tagger.setMaxCounts(_max_runs)

					PulseGenerator().Sequence(_sequence)

					time.sleep(_wait_time)
					
					_runs = tagger.getCounts()
					
					_iter = 0

					while _runs < _max_runs:
						_iter += 1
						time.sleep(_wait_time / 20)

						_runs = tagger.getCounts()
						
						if _iter >= 20:
							logging.getLogger().warn('Timed out. Insufficient measurement runs collected for references')
							break

					self.count_data[-4:] += tagger.getData()
					self.runs_number[-4:] += _runs

				self.trait_property_changed('count_data', self.count_data)
				self.run_time += time.time() - t_start

			if self.run_time < self.stop_time:
				self.state = 'idle'
			else:
				self.state = 'done'
				
		except Exception as exp:
			traceback.print_exc()
			logging.getLogger().error(str(exp))
			self.state = 'error'
		finally:
			del tagger
			PulseGenerator().Light()
			Microwave().setOutput(None, self.mw1_frequency)
			MicrowaveB().setOutput(None, self.rf_begin)

	get_set_items = Pulsed.get_set_items + [\
		'measurement_type',
		'mw1_power', 'mw1_frequency', 't_mw1_pi2', 't_mw1_pi', 't_mw1_3pi2',
		'mw2_power', 'rf_begin', 'rf_end', 'rf_delta', 't_mw2_pi',
		'references', 'references_tau_mix',
		'tau', 'frequencies', 'sequence',
		'laser', 'wait', 'seconds_per_point',
		'count_data', 'runs_number',
		'randomize', 'cap_runs', 'cap_runs_margin'
	]

	traits_view = View(\
		VGroup(\
			HGroup(\
				Item('submit_button', show_label=False),
				Item('remove_button', show_label=False),
				Item('resubmit_button', show_label=False),
				Item('priority', width= -40),
				Item('state', style='readonly'),
				Item('run_time', style='readonly'),
				Item('stop_time', width= -40),
			),
			Tabbed(\
				VGroup(\
					HGroup(\
						Item('mw1_power', width= -40, enabled_when='state != "run"'),
						Item('mw1_frequency', width= -120, enabled_when='state != "run"'),
						Item('t_mw1_pi2', width= -40, enabled_when='state != "run"'),
						Item('t_mw1_pi', width= -40, enabled_when='state != "run"'),
						Item('t_mw1_3pi2', width= -40, enabled_when='state != "run"'),
						Item('tau', width= -40, enabled_when='state != "run"'),
					),
					HGroup(\
						Item('mw2_power', width= -40, enabled_when='state != "run"'),
						Item('rf_begin', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x), enabled_when='state != "run"'),
						Item('rf_end', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x), enabled_when='state != "run"'),
						Item('rf_delta', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x), enabled_when='state != "run"'),
						Item('t_mw2_pi', width= -40, enabled_when='state != "run"'),
					),
					HGroup(\
						Item('references', width= -40, enabled_when='state != "run"'),
						Item('references_tau_mix', width= -80, enabled_when='state != "run"'),
					),
					label='Parameter'
				),
				VGroup(\
					HGroup(\
						Item('laser', width= -80, enabled_when='state != "run"'),
						Item('wait', width= -80, enabled_when='state != "run"'),
						Item('record_length', width= -80, enabled_when='state != "run"'),
						Item('bin_width', width= -80, enabled_when='state != "run"'),
					),
					HGroup(\
						Item('seconds_per_point', width= -120, enabled_when='state != "run"'),
						Item('sweeps_per_point', width= -120, style='readonly'),
					),
					label='Settings'
				),
				VGroup(\
					HGroup(\
						Item('randomize', width= -40, enabled_when='state != "run"'),
					),
					HGroup(\
						Item('cap_runs', width= -40, enabled_when='state != "run"'),
						Item('cap_runs_margin', width= -40, enabled_when='state != "run"')
					),
					label='Others',
				),
			),
		),
		title='DEER 3pi/2',
	)

class DEERRabi3pi2(Pulsed):

	measurement_type = 'deerrabi_3pi2'

	mw1_power = Range(low= -100., high=25., value=-20, desc='MW1 Power [dBm]', label='MW1 Power [dBm]', mode='text', auto_set=False, enter_set=True)
	mw1_frequency = Range(low=1., high=20.e9, value=2.87e9, desc='MW1 Frequency [Hz]', label='MW1 Frequency [Hz]', mode='text', auto_set=False, enter_set=True)
	t_mw1_pi2 = Range(low=1., high=100000., value=83., desc='length of pi/2 pulse of mw1 [ns]', label='mw1 pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
	t_mw1_pi = Range(low=1., high=100000., value=166., desc='length of pi pulse of mw1 [ns]', label='mw1 pi [ns]', mode='text', auto_set=False, enter_set=True)
	t_mw1_3pi2 = Range(low=1., high=100000., value=250., desc='length of 3pi/2 pulse of mw1 [ns]', label='mw1 3pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
	int_tau = Range(low=0., high=5000000., value=300., desc='Interaction Tau [ns]', label='Interaction Tau [ns]', mode='text', auto_set=False, enter_set=True)
	mw2_power = Range(low= -100., high=25., value=-20, desc='MW2 Power [dBm]', label='MW2 Power [dBm]', mode='text', auto_set=False, enter_set=True)
	mw2_frequency = Range(low=1., high=1000e6, value=200e6, desc='MW2 Frequency [Hz]', label='MW2 frequency [Hz]', mode='text', auto_set=False, enter_set=True)
	tau_begin = Range(low=0, high=1e5, value=50, desc='Start Tau [ns]', label='Begin [ns]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
	tau_end = Range(low=1, high='int_tau', value=500, desc='Stop Tau [ns]', label='End [ns]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
	tau_delta = Range(low=1, high=1e4, value=10, desc='Delta Tau [ns]', label='Delta [ns]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))

	laser = Range(low=1., high=10000., value=3000., desc='Laser [ns]', label='Laser [ns]', mode='text', auto_set=False, enter_set=True)
	wait = Range(low=1., high=10000., value=1000., desc='Wait [ns]', label='Wait [ns]', mode='text', auto_set=False, enter_set=True)

	run_time = Float(value=0.0) 

	randomize = Bool(False)
	randomize_interval = Float(10.0)

	variable_tau = Bool(False)

	tau = Array(value=np.array((0., 1.)))  

	def generate_sequence(self, randomize=False, rand_mapping=[]):
		_sequence = []
		if randomize:
			for _type, _ind in rand_mapping:
				_rf_t = self.tau[_ind]

				if self.variable_tau:
					_i_tau = self.int_tau - _rf_t
					if _i_tau < 0:
						raise Exception
				else:
					_i_tau = self.int_tau

				if _type == 0:
					_sequence.extend([ (['mw_a'], self.t_mw1_pi2), ([], _i_tau), (['mw_b'], _rf_t), (['mw_a'], self.t_mw1_pi), ([], self.int_tau), (['mw_a'], self.t_mw1_pi2), (['laser', 'aom'], self.laser), ([], self.wait) ])
				elif _type == 1:
					_sequence.extend([ (['mw_a'], self.t_mw1_pi2), ([], _i_tau), (['mw_b'], _rf_t), (['mw_a'], self.t_mw1_pi), ([], self.int_tau), (['mw_a'], self.t_mw1_3pi2), (['laser', 'aom'], self.laser), ([], self.wait) ])

			_sequence.append((['sequence'], 100))

			return _sequence

		else:
			_3pi2_sequence = []
			for _rf_t in self.tau:
				if self.variable_tau:
					_i_tau = self.int_tau - _rf_t
					if _i_tau < 0:
						raise Exception
				else:
					_i_tau = self.int_tau

				_sequence.extend([ (['mw_a'], self.t_mw1_pi2), ([], _i_tau), (['mw_b'], _rf_t), (['mw_a'], self.t_mw1_pi), ([], self.int_tau), (['mw_a'], self.t_mw1_pi2), (['laser', 'aom'], self.laser), ([], self.wait) ])
				_3pi2_sequence.extend([ (['mw_a'], self.t_mw1_pi2), ([], _i_tau), (['mw_b'], _rf_t), (['mw_a'], self.t_mw1_pi), ([], self.int_tau), (['mw_a'], self.t_mw1_3pi2), (['laser', 'aom'], self.laser), ([], self.wait) ])
			
			_sequence.extend(_3pi2_sequence)

			_sequence.append((['sequence'], 100))

		return _sequence

	def apply_parameters(self):
		"""Apply the current parameters and decide whether to keep previous data."""

		self.tau = np.arange(self.tau_begin, self.tau_end + self.tau_delta, self.tau_delta)
		n_bins = int(self.record_length / self.bin_width)
		time_bins = self.bin_width * np.arange(n_bins)

		sequence = self.generate_sequence()

		if not (self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins)): # if the sequence and time_bins are the same as previous, keep existing data
			self.count_data = np.zeros((2 * self.tau.size, n_bins))
			self.run_time = 0.0

		self.sequence = sequence 
		self.time_bins = time_bins
		self.n_bins = n_bins
		self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.

	def _run(self):
		"""Acquire data."""
		
		try: # try to run the acquisition from start_up to shut_down
			self.state = 'run'
			self.apply_parameters()
			PulseGenerator().Night()
			Microwave().setOutput(self.mw1_power, self.mw1_frequency)
			MicrowaveB().setOutput(self.mw2_power, self.mw2_frequency)

			_len = self.tau.size

			tagger = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), _len * 2, 0, 2, 3)

			if not self.randomize:
				PulseGenerator().Sequence(self.sequence)

			while self.run_time < self.stop_time:

				if self.thread.stop_request.isSet():
					break

				if self.randomize:
					_rand_mapping = list( zip( [0] * _len, range(0, _len) )) + list( zip( [1] * _len, range(0, _len) ))
					random.shuffle(_rand_mapping)
					logging.getLogger().debug(str(_rand_mapping))

					_sequence = self.generate_sequence(True, _rand_mapping)
					PulseGenerator().Sequence(_sequence)
					time.sleep(0.0001)
					tagger.clear()

				t_start = time.time()

				if self.randomize:
					self.thread.stop_request.wait(self.randomize_interval)
				else:
					self.thread.stop_request.wait(1)

				_data = tagger.getData()

				if self.randomize:
					_sorted_data = np.zeros((_len * 2, self.n_bins))

					for _old_ind, _mapping in enumerate(_rand_mapping):
						_pos = _mapping[0] * _len + _mapping[1]
						logging.getLogger().debug('Data Insertion Position: ' + str(_pos))
				
						_sorted_data[_pos] = _data[_old_ind]

					self.count_data += _sorted_data

				else:
					self.count_data += _data

				self.trait_property_changed('count_data', self.count_data)

				self.run_time += time.time() - t_start

			if self.run_time < self.stop_time:
				self.state = 'idle'
			else:
				self.state = 'done'
				
		except Exception as exp:
			traceback.print_exc()
			logging.getLogger().error(str(exp))
			self.state = 'error'
		finally:
			del tagger
			PulseGenerator().Light()
			Microwave().setOutput(None, self.mw1_frequency)
			MicrowaveB().setOutput(None, self.mw2_frequency)

	get_set_items = Pulsed.get_set_items + [\
		'measurement_type',
		'mw1_power', 'mw1_frequency', 't_mw1_pi2', 't_mw1_pi', 't_mw1_3pi2', 'int_tau',
		'mw2_power', 'mw2_frequency', 'tau_begin', 'tau_end', 'tau_delta', 'tau',
		'laser', 'wait', 'count_data', 'sequence', 'randomize', 'randomize_interval', 'variable_tau'
	]

	traits_view = View(\
		VGroup(\
			HGroup(\
				Item('submit_button', show_label=False),
				Item('remove_button', show_label=False),
				Item('resubmit_button', show_label=False),
				Item('priority', width= -40),
				Item('state', style='readonly'),
				Item('run_time', style='readonly'),
				Item('stop_time', width= -40),
			),
			Tabbed(\
				VGroup(\
					HGroup(\
						Item('mw1_power', width= -40, enabled_when='state != "run"'),
						Item('mw1_frequency', width= -120, enabled_when='state != "run"'),
						Item('t_mw1_pi2', width= -40, enabled_when='state != "run"'),
						Item('t_mw1_pi', width= -40, enabled_when='state != "run"'),
						Item('t_mw1_3pi2', width= -40, enabled_when='state != "run"'),
						Item('int_tau', width= -40, enabled_when='state != "run"'),
					),
					HGroup(\
						Item('mw2_power', width= -40, enabled_when='state != "run"'),
						Item('mw2_frequency', width= -120, enabled_when='state != "run"'),
						Item('tau_begin', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x), enabled_when='state != "run"'),
						Item('tau_end', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x), enabled_when='state != "run"'),
						Item('tau_delta', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x), enabled_when='state != "run"'),
					),
					label='Parameter'
				),
				VGroup(\
					HGroup(\
						Item('laser', width= -80, enabled_when='state != "run"'),
						Item('wait', width= -80, enabled_when='state != "run"'),
						Item('record_length', width= -80, enabled_when='state != "run"'),
						Item('bin_width', width= -80, enabled_when='state != "run"'),
					),
					label='Settings'
				),
				VGroup(\
					HGroup(\
						Item('randomize', width= -40, enabled_when='state != "run"'),
						Item('randomize_interval', width=-80, enabled_when='state != "run"'),
					),
					HGroup(\
						Item('variable_tau', width= -40, enabled_when='state != "run"'),
					),
					label='Others',
				),
			),
		),
		title='DEER Rabi 3pi/2',
	)
