import numpy as np

# Enthought library imports
from enable.api import Component, ComponentEditor
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

# Chaco imports
from chaco.api import ArrayDataSource, ArrayPlotData, BarPlot, DataRange1D, LabelAxis, LinearMapper, OverlayPlotContainer, Plot, PlotAxis, PlotLabel, cbrewer as COLOR_PALETTE

from .definitions import color_map

# Based on the Tornado plot example from Brennan Williams
class Histogram(HasTraits):

	def __init__(self, dual=False):
		super(Histogram, self).__init__()

		self.add_trait("plot", OverlayPlotContainer(padding=0))

		self.dual = dual

		# Dummy Value
		_label = ["0", "1", "2", "3"]
		_label_index = [0, 1, 2, 3]

		self.x_data = ArrayDataSource(np.array([0, 1, 2, 3]))

		self.y1_data = ArrayDataSource(np.array([0, 1, 1, 0]))

		if dual:
			self.y2_data = ArrayDataSource(np.array([0, 2, 2, 0]))

		self.index_range = DataRange1D(self.x_data, low_setting="auto", high_setting="auto", stretch_data=False)

		self.index_mapper = LinearMapper(range=self.index_range)
		
		if not dual:
			self.value_range = DataRange1D(self.y1_data, low_setting="auto", high_setting="auto", stretch_data=False)
		else:
			self.value_range = DataRange1D(self.y1_data, self.y2_data, low_setting="auto", high_setting="auto", stretch_data=False)

		self.value_mapper = LinearMapper(range=self.value_range)

		AXIS_DEFAULTS = {
			'axis_line_weight': 5,
			'tick_weight': 5,
			'tick_label_font': 'modern 12',
			'title_font': 'modern 16',
			'tick_out': 0,
			'tick_in': 10
		}

		self.plot1 = BarPlot(
			index=self.x_data, value=self.y1_data, value_mapper=self.value_mapper, index_mapper=self.index_mapper,
			alpha=0.5, line_color="black", orientation="h", fill_color=color_map['y1'], bar_width=1, antialias=False, padding=8, padding_left=64, padding_bottom=36
		)
		
		self.plot.add(self.plot1)
		
		if self.dual:
			self.plot2 = BarPlot(
				index=self.x_data, value=self.y2_data, value_mapper=self.value_mapper, index_mapper=self.index_mapper,
				alpha=0.5, line_color="black", orientation="h", fill_color=color_map['y2'], bar_width=1, antialias=False, padding=8, padding_left=64, padding_bottom=36
			)
		
			self.plot.add(self.plot2)

		self.x_axis = LabelAxis(labels=_label, positions=_label_index, component=self.plot1, orientation="bottom", ensure_labels_bounded=True, tick_label_position="outside", title="Photon Counts", tight_bounds=True, **AXIS_DEFAULTS)
		self.y_axis = PlotAxis(mapper=self.value_mapper, component=self.plot1, orientation="left", ensure_labels_bounded=True, tick_label_position="outside", title="Occurrence", tight_bounds=True, **AXIS_DEFAULTS)

		self.plot1.underlays.append(self.x_axis)
		self.plot1.underlays.append(self.y_axis)

	def update_data(self, name, data):
		if name == "x":
			_min = data[0]
			_max = data[-1]

			_index = []
			_data = np.zeros(len(data) + 4)

			_data[2:-2] = data
			_data[0] = _min - 2
			_data[1] = _min - 1
			_data[-2] = _max + 1
			_data[-1] = _max + 2

			_index.append("")
			_index.append("< %s" %_min)
			for i in data:
				_index.append(str(i))
			_index.append("> %s" %_max)
			_index.append("")

			self.x_axis.labels = _index
			self.x_axis.positions = _data

			self.x_data.set_data(_data)

		if name == "y1":
			_data = np.zeros(len(data) + 2)

			_data[1:-1] = data

			self.y1_data.set_data(_data)
		
		if self.dual and name == "y2":
			_data = np.zeros(len(data) + 2)

			_data[1:-1] = data

			self.y2_data.set_data(_data)

	traits_view = View(
		Item("plot", editor=ComponentEditor(), show_label=False),
		resizable=True,
		width=800,
		height=600,
	)

class LinePlot(HasTraits):

	def __init__(self, dual=False, x_title="Tau [ns]", y_title="Signal [a.u]"):
		super(LinePlot, self).__init__()

		self.dual = dual

		self.data = ArrayPlotData(x=np.array((0,1)), y1=np.array((0,0)), y2=np.array((0,0)))

		self.active_plot = []

		self.add_trait("plot", Plot(self.data, padding=8, padding_left=64, padding_bottom=36))

		self.plot.plot(('x','y1'), color=color_map['y1'], line_width=2, id='0', name='y1')
		self.active_plot.append('y1')
		if self.dual:
			self.plot.plot(('x','y2'), color=color_map['y2'], line_width=2, id='0', name='y2')
			self.active_plot.append('y2')

		self.plot.bgcolor = color_map['background']
		self.plot.x_grid = None
		self.plot.y_grid = None
		self.plot.index_axis.title = x_title
		self.plot.value_axis.title = y_title

		self.line_label = PlotLabel(text='', hjustify='left', vjustify='bottom', position=[64, 128])
		self.plot.overlays.append(self.line_label)

	def update_data(self, name, data):
		if name == "x":
			self.data.set_data("x", data)
		
		else:
			_color = color_map.get(name, None)

			if isinstance(data, (list, np.ndarray)):
				self.data.set_data(name, data)
				if name not in self.active_plot:
					self.active_plot.append(name)
					self.plot.plot(('x', name), color=_color, line_width=2, id='0', name=name)

			elif isinstance(data, (int, float)):
				_len = self.data.get_data('x').size
				_data = numpy.ones(_len) * data
				
				self.data.set_data(name, _data)
				if name not in self.active_plot:
					self.active_plot.append(name)
					self.plot.plot(('x', name), color=_color, line_width=2, id='0', name=name)

			elif data == None and name in self.active_plot:
				self.active_plot.remove(name)
				self.plot.delplot(name)
				self.data.del_data(name)

	traits_view = View(
		Item("plot", editor=ComponentEditor(), show_label=False),
		resizable=True,
		width=800,
		height=600,
	)
