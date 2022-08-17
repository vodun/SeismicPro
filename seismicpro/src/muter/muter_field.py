from .muter import Muter
from ..field import ValuesAgnosticField, VFUNCFieldMixin


class MuterField(ValuesAgnosticField, VFUNCFieldMixin):
    item_class = Muter

    def construct_item(self, items, weights, coords):
        return self.item_class.from_muters(items, weights, coords=coords)

    @classmethod
    def from_refractor_velocity_field(cls, field, delay=0, velocity_reduction=0):
        items = [cls.item_class.from_refractor_velocity(item, delay=delay, velocity_reduction=velocity_reduction)
                 for item in field.item_container.values()]
        return cls(items, survey=field.survey, is_geographic=field.is_geographic)

    @classmethod
    def from_stacking_velocity_field(cls, field, stretch_factor=0.65):
        items = [cls.item_class.from_stacking_velocity(item, stretch_factor=stretch_factor)
                 for item in field.item_container.values()]
        return cls(items, survey=field.survey, is_geographic=field.is_geographic)
