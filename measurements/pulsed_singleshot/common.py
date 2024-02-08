import numpy as np
import logging

from traits.api import HasTraits, Range
from traitsui.api import View, Item, HGroup

from tools.utility import GetSetItemsMixin


class DataPointsSelector(HasTraits, GetSetItemsMixin):
    pass


class LinearPoints(DataPointsSelector):

    get_set_items = ["start", "end", "delta"]

    def __init__(self, range, unit, **kwargs):

        _lower = range[0]
        _upper = range[1]
        self._lower = _lower
        self._upper = _upper
        self.unit = unit

        _type_low = type(_lower)
        _type_high = type(_upper)

        if _type_low != _type_high:
            logging.getLogger().Warning("Datapoints: Limits have different types")

        super(LinearPoints, self).__init__()

        self.add_trait(
            "start",
            Range(
                low=_lower,
                high=_upper,
                label="Start [%s]" % unit,
                mode="text",
                auto_set=False,
                enter_set=True,
                **kwargs
            ),
        )
        self.add_trait(
            "end",
            Range(
                low=_lower,
                high=_upper,
                label="End [%s]" % unit,
                mode="text",
                auto_set=False,
                enter_set=True,
                **kwargs
            ),
        )
        self.add_trait(
            "delta",
            Range(
                low=_type_low(0),
                high=_upper,
                label="Delta [%s]" % unit,
                mode="text",
                auto_set=False,
                enter_set=True,
                **kwargs
            ),
        )

    def get_datapoints(self):
        return np.arange(self.start, self.end + self.delta, self.delta)

    traits_view = View(
        HGroup(
            Item(name="start", show_label=True),
            Item(name="end", show_label=True),
            Item(name="delta", show_label=True),
        ),
    )
