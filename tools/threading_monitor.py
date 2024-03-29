from traits.api import (
    HasTraits,
    HasStrictTraits,
    Instance,
    Str,
    List,
)
from traitsui.api import (
    View,
    Item,
)
from traitsui.api import TableEditor, ObjectColumn


from tools.cron import CronDaemon


class Events(HasStrictTraits):

    method = Str("method")
    dependencies = Str("module.method")
    address = Str("0x00000000")

    def __init__(self, event):
        raw_repr = event.__dict__["action"].__repr__()
        raw_repr = raw_repr.replace("<", "").replace(">", "")
        after_bound_method = raw_repr.split("bound method ")[1]
        before_of = after_bound_method.split(" of ")[0]
        after_of = after_bound_method.split(" of ")[1]
        before_objectat = after_of.split(" object at ")[0]
        after_objectat = after_of.split(" object at ")[1]

        self.method = before_of
        self.dependencies = before_objectat
        self.address = after_objectat

    traits_view = View("method", "dependencies", "address", buttons=["OK", "Cancel"])


class EventMonitor(HasTraits):

    Events_list = List(Instance(Events))
    Repr_list = List()

    def __init__(self):
        Events_list = []
        Repr_list = []
        for event in CronDaemon().events:
            event_processed = Events(event)
            Events_list.append(event_processed)
            li = [
                event_processed.method,
                event_processed.dependencies,
                event_processed.address,
            ]
            Repr_list.append(li)
        self.Events_list = Events_list
        self.Repr_list = Repr_list

    table_editor = TableEditor(
        columns=[
            ObjectColumn(name="method", width=0.4),
            ObjectColumn(name="dependencies", width=0.2),
            ObjectColumn(name="address", width=0.4),
        ],
        editable=False,
        sortable=True,
        row_factory=Events,
    )

    traits_view = View(
        Item("Events_list", show_label=False, editor=table_editor),
        title="Events Monitor",
        resizable=True,
        width=400,
        height=400,
    )
