import numpy as np

from traits.api import Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group, Label

import logging
import time

import random

from hardware.api import PulseGenerator, TimeTagger, Microwave
import hardware.api as ha
from tools.emod import ManagedJob

from tools.utility import GetSetItemsMixin

from pulsed_rabi import Rabi
from pulsed import Pulsed

class Hahn3pi2(Rabi):	
	"""Defines a Hahn-Echo measurement with both pi/2 and 3pi/2 readout pulse."""

	measurement_type = 'hahn_3pi2'

	t_pi2 = Range(low=1., high=100000., value=1000., desc='pi/2 pulse length', label='pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
	t_pi = Range(low=1., high=100000., value=1000., desc='pi pulse length', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
	t_3pi2 = Range(low=1., high=100000., value=1000., desc='3pi/2 pulse length', label='3pi/2 [ns]', mode='text', auto_set=False, enter_set=True)

	tau_splitting = Range(low=1., high=10000., value=1.0, label='Tau Splitting [ns]', mode='text', auto_set=False, enter_set=True)

	def apply_parameters(self):

		_norm_tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)

		if self.tau_splitting:
			self.tau = np.zeros(len(_norm_tau) * 3 - 2)

			self.tau[0] = _norm_tau[0]
			self.tau[2::3] = _norm_tau[1:]
			self.tau[1::3] = _norm_tau[1:] - self.tau_splitting
			self.tau[3::3] = _norm_tau[1:] + self.tau_splitting
		else:
			self.tau = _norm_tau

		Pulsed.apply_parameters(self)

	def generate_sequence(self, rand=False, order=[]):
		MW = self.switch
		tau = self.tau
		laser = self.laser
		wait = self.wait
		t_pi2 = self.t_pi2
		t_pi = self.t_pi
		t_3pi2 = self.t_3pi2

		_max_tau = tau[-1]

		sequence = []
		_sequence_3pi2 = []

		if not rand:
			_end_ind = len(tau) - 1

			for i, t in enumerate(tau):
				_wait = self.wait
				if self.variable_wait and i < _end_ind:
					_wait += abs(_max_tau - tau[i + 1])

				sequence += [ ([MW], t_pi2), ([], 0.5 * t), ([MW], t_pi), ([], 0.5 * t), ([MW], t_pi2), (['laser', 'aom'], laser), ([], _wait) ]
				_sequence_3pi2 += [ ([MW], t_pi2), ([], 0.5 * t), ([MW], t_pi), ([], 0.5 * t), ([MW], t_3pi2), (['laser', 'aom'], laser), ([], _wait) ]

			sequence.extend(_sequence_3pi2)
			
		else:
			for type, i in order:
				if type == 0:
					_mw_t = t_pi2
				elif type == 1:
					_mw_t = t_3pi2

				t = tau[i]

				_wait = self.wait
				if self.variable_wait and i < _end_ind * 2 + 1:
					_next_mw_type = order[i + 1][0]
					if _next_mw_type == 0:
						_next_mw_t = t_pi2
					elif _next_mw_type == 1:
						_next_mw_t = t_3pi2
					_next_t = order[i + 1][1]

					_wait = _wait + _max_tau - _next_t + _next_mw_t

				sequence += [ ([MW], t_pi2), ([], 0.5 * t), ([MW], t_pi), ([], 0.5 * t), ([MW], _mw_t), (['laser', 'aom'], laser), ([], _wait) ]

		sequence += [ (['sequence'], 100 ) ]

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
					_rand_mapping = list( zip( [0] * _len, range(0, _len) )) + list( zip( [1] * _len, range(0, _len) ))
					random.shuffle(_rand_mapping)
					_sequence = self.generate_sequence(True, _rand_mapping)
					PulseGenerator().Sequence(_sequence)
					tagger_0.clear()
					tagger_1.clear()

				if PulseGenerator().checkUnderflow():
					logging.getLogger().info('Underflow in pulse generator.')
					print('Underflow')
					PulseGenerator().Night()
					if self.randomize:
						_rand_mapping = list( zip( [0] * _len, range(0, _len) )) + list( zip( [1] * _len, range(0, _len) ))
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
					sorted_data = np.zeros((_len * 2, self.n_bins))
					for _old_ind, _mapping in enumerate(_rand_mapping):
						_pos = _mapping[1] + _mapping[0] * _len

						logging.getLogger().debug('Data Insertion Position : ' + str(_pos))

						sorted_data[_pos] = currentcountdata[_old_ind]

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
			PulseGenerator().Light()

		except: # if anything fails, log the exception and set the state
			logging.getLogger().exception('Something went wrong in pulsed loop.')
			self.state = 'error'

	traits_view = View(\
		VGroup(\
			HGroup(\
				Item('submit_button', show_label=False),
				Item('remove_button', show_label=False),
				Item('resubmit_button', show_label=False),
				Item('priority'),
				Item('state', style='readonly'),
				Item('run_time', style='readonly', format_str='%.f'),
				Item('stop_time'),
			),
			Tabbed(\
				VGroup(\
					HGroup(\
						Item('switch', enabled_when='state != "run"'),
						Item('frequency', width= -80, enabled_when='state != "run"'),
						Item('power', width= -80, enabled_when='state != "run"'),
					),
					HGroup(\
						Item('t_pi2', width= -80, enabled_when='state != "run"'),
						Item('t_pi', width= -80, enabled_when='state != "run"'),
						Item('t_3pi2', width= -80, enabled_when='state != "run"'),
					),
					HGroup(\
						Item('tau_begin', width= -80, enabled_when='state != "run"'),
						Item('tau_end', width= -80, enabled_when='state != "run"'),
						Item('tau_delta', width= -80, enabled_when='state != "run"'),
                    ),
                    HGroup(\
						Item('tau_splitting', width=-80, enabled_when='state != "run"'),
					),
					label='Parameters'
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
		title='Hahn-Echo Measurement with both pi/2 and 3pi/2 readout pulse',
	)
	
	get_set_items = Rabi.get_set_items + ['t_pi2', 't_pi', 't_3pi2']
