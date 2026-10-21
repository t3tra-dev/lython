# WHAT: subclasses whose base is written through a relative import
# (`from . import shape` / `class Square(shape.Shape)`, `from .shape import
# Shape`) and through a package's re-export (`from a_package_of_shapes import
# Shape`, a name that differs from the module `shape.py` only in case), held as
# the base and asked for their name; and `isinstance` of each.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the base's method ran for
# every subclass -- ['shape', 'shape', ...] -- and nothing was refused, so only
# the printed names show which body the call reached.
import a_package_of_shapes
from a_package_of_shapes import shape
import a_subclass_through_a_relative_import_runs_its_override_lib as other

shapes: list[a_package_of_shapes.Shape] = [
    a_package_of_shapes.Shape(),
    a_package_of_shapes.Square(),
    a_package_of_shapes.Circle(),
    other.Triangle(),
    other.Hexagon(),
]
print([shape.describe(s) for s in shapes])
print([isinstance(s, a_package_of_shapes.Shape) for s in shapes])
print([isinstance(s, a_package_of_shapes.Square) for s in shapes])
