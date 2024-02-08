import numpy as np

from traits.api import Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group, Label

import logging
import time

import random

from hardware.api import PulseGenerator, TimeTagger, Microwave
import hardware.api as ha
from pulsed import Pulsed
from tools.emod import ManagedJob

from tools.utility import GetSetItemsMixin


class Rabi( Pulsed ):
	"""Defines a Rabi measurement."""

	measurement_type = 'rabi'

	switch = Enum( 'mw_x', 'mw_y', 'mw_b', 'mw_c', desc='switch to use for different microwave source', label='switch' )
	frequency = Range(low=1, high=20e9, value=2.87e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True)
	power = Range(low=-100., high=12.5, value=-20, desc='microwave power', label='power [dBm]', mode='text', auto_set=False, enter_set=True)

	tau_begin = Range(low=0., high=1e8, value=0, desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
	tau_end = Range(low=1., high=1e8, value=1000., desc='tau end [ns]', label='tau end [ns]',	 mode='text', auto_set=False, enter_set=True)
	tau_delta = Range(low=1., high=1e6, value=10., desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
	laser = Range(low=1., high=100000., value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
	wait = Range(low=0., high=100000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
	
	randomize = Bool(True)
	randomize_interval = Float(5.0)

	# Variable wait time to keep the interval between laser pulses constants
	variable_wait = Bool(False)

	tau = Array( value=np.array((0.,1.)) )

	time_bins = Array(value=np.array((0, 1)))
	sequence = Instance(list, factory=list)
	keep_data = Bool(False)
	def apply_parameters(self):
		"""Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
		self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)

		Pulsed.apply_parameters(self)
		
	def start_up(self):
		PulseGenerator().Night()
		if self.switch=='mw_a':
			ha.MicrowaveA().setOutput(self.power, self.frequency)
		elif self.switch=='mw_x' or self.switch=='mw_y':
			ha.MicrowaveA().setOutput(self.power, self.frequency)
		elif self.switch=='mw_b':
			ha.MicrowaveB().setOutput(self.power, self.frequency)
		elif self.switch=='mw_c':
			ha.MicrowaveC().setOutput(self.power, self.frequency)
			
	def shut_down(self):
		PulseGenerator().Light()
		if self.switch=='mw_a':
			ha.MicrowaveA().setOutput(None, self.frequency)
		elif self.switch=='mw_x' or self.switch=='mw_y':
			ha.MicrowaveA().setOutput(None, self.frequency)
		elif self.switch=='mw_b':
			ha.MicrowaveB().setOutput(None, self.frequency)
		elif self.switch=='mw_c':
			ha.MicrowaveC().setOutput(None, self.frequency)

	def generate_sequence(self, rand=False, order=[]):
		MW = self.switch
		tau = self.tau
		laser = self.laser
		wait = self.wait
		sequence = []

		_max_tau = tau[-1]
		_end_ind = len(tau) - 1

		if not rand:
			for i, t in enumerate(tau):
				_wait = self.wait
				if self.variable_wait and i < _end_ind:
					_wait += abs(_max_tau - tau[i + 1])

				sequence += [ ([MW],t), (['laser', 'aom'], laser), ([], _wait) ]
			sequence += [ (['sequence'], 100 ) ]
		else:
			for i, j in enumerate(order):
				t = tau[j]
				_wait = self.wait
				if self.variable_wait and i < _end_ind:
					_wait += abs(_max_tau - tau[order[i + 1]])

				sequence += [ ([MW], t), (['laser','aom'], laser), ([], _wait) ]
			sequence += [ (['sequence'], 100 ) ]

		logging.getLogger().debug('Truncated Sequence ' + str(sequence[:20]))

		return sequence

	def _run(self):
		"""Acquire data."""

		try: # try to run the acquisition from start_up to shut_down
			self.state = 'run'
			self.apply_parameters()
			if self.run_time >= self.stop_time:
				logging.getLogger().debug('Runtime larger than stop_time. Returning')
				self.state = 'done'
				return

			self.count_data = self.old_count_data

			self.start_up()
			PulseGenerator().Night()

			#tagger_0 = TimeTagger.Pulsed(int(self.n_bins), int(np.round(self.bin_width * 1000)), int(self.n_laser), Int(0), Int(2), Int(3))
			#tagger_1 = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), self.n_laser, Int(1), Int(2), Int(3))

			tagger_0 = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), self.n_laser, 0, 2, 3)
			tagger_1 = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), self.n_laser, 1, 2, 3)

			if not self.randomize:
				_sequence = self.sequence
				PulseGenerator().Sequence(_sequence)
			else:
				_len = self.tau.size

			if PulseGenerator().checkUnderflow():
				logging.getLogger().info('Underflow in pulse generator.')
				PulseGenerator().Night()
				if not self.randomize:
					PulseGenerator().Sequence(_sequence)

			while self.run_time < self.stop_time:
				start_time = time.time()

				if self.randomize:
					_rand_mapping = range(0, _len)
					random.shuffle(_rand_mapping)
					_sequence = self.generate_sequence(True, _rand_mapping)

					PulseGenerator().Sequence(_sequence)

					tagger_0.clear()
					tagger_1.clear()

				if PulseGenerator().checkUnderflow():
					logging.getLogger().info('Underflow in pulse generator.')

					PulseGenerator().Night()
					if self.randomize:
						_rand_mapping = range(0, _len)
						random.shuffle(_rand_mapping)
						_sequence = self.generate_sequence(True, _rand_mapping)
						PulseGenerator().Sequence(_sequence)

						tagger_0.clear()
						tagger_1.clear()

					else:
						PulseGenerator().Sequence(_sequence)

				if self.randomize:
					self.thread.stop_request.wait(self.randomize_interval)
				else:
					self.thread.stop_request.wait(1)

				currentcountdata0 = tagger_0.getData() 
				currentcountdata1 = tagger_1.getData()
				currentcountdata = currentcountdata1 + currentcountdata0

				if self.randomize:
					sorted_data = np.zeros((_len, self.n_bins))
					for _old_ind, _new_ind in enumerate(_rand_mapping):
						sorted_data[_new_ind] = currentcountdata[_old_ind]

					self.count_data = self.count_data + sorted_data

				else:
					self.count_data = self.old_count_data + currentcountdata

				self.run_time += time.time() - start_time

				if self.thread.stop_request.isSet():
					logging.getLogger().debug('Caught stop signal. Exiting.')
					break

			if self.run_time < self.stop_time:
				self.state = 'idle'
			else:
				self.state = 'done'

			del tagger_0
			del tagger_1

			self.shut_down()

		except: # if anything fails, log the exception and set the state
			logging.getLogger().exception('Something went wrong in pulsed loop.')
			self.state = 'error'

	get_set_items = Pulsed.get_set_items + ['measurement_type', 'frequency','power','switch','tau_begin','tau_end','tau_delta','laser','wait','tau', 'randomize', 'randomize_interval', 'variable_wait']
	get_set_order = ['tau','time_bins','count_data']

	traits_view = View(\
		VGroup(\
			HGroup(\
				Item('submit_button', show_label=False),
				Item('remove_button',   show_label=False),
				Item('resubmit_button', show_label=False),
				Item('priority'),
				Item('state', style='readonly'),
				Item('run_time', style='readonly',format_str='%.f'),
				Item('stop_time'),
			),
			Tabbed(\
				VGroup(\
					HGroup(\
						Item('switch', enabled_when='state != "run"'),
						Item('frequency', width=-80, enabled_when='state != "run"'),
						Item('power', width=-80, enabled_when='state != "run"'),
					),
					HGroup(\
						Item('tau_begin', width=-80, enabled_when='state != "run"'),
						Item('tau_end', width=-80, enabled_when='state != "run"'),
						Item('tau_delta', width=-80, enabled_when='state != "run"'),
					),
					label='Parameters'
				),
				VGroup(\
					HGroup(\
						Item('laser', width=-80, enabled_when='state != "run"'),
						Item('wait', width=-80, enabled_when='state != "run"'),
						Item('record_length', width=-80, enabled_when='state != "run"'),
						Item('bin_width', width=-80, enabled_when='state != "run"'),
					),
					label='Settings'
				),
				VGroup(\
					HGroup(\
						Item('randomize', width=-80, enabled_when='state != "run"'),
						Item('randomize_interval', width=-80, enabled_when='state != "run"'),
					),
					HGroup(\
						Item('variable_wait', width=-80, enabled_when='state != "run"'),
					),
					label='Others'
				),
			),
		),
		title='Rabi',
	)
