# A module outside the package, naming the base through the package's own
# re-export and through the package.
from a_package_of_shapes import Shape
import a_package_of_shapes


class Triangle(Shape):
    def name(self) -> str:
        return "triangle"


class Hexagon(a_package_of_shapes.Shape):
    def name(self) -> str:
        return "hexagon"
