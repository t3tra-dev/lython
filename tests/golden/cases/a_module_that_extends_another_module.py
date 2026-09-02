# What: the middle module of `a_subclass_declared_in_a_third_module`. It is a
# case of its own because the golden runner globs every .py here -- running it
# alone declares two classes and prints nothing, which is what its empty
# expectation says. Both import spellings appear on purpose: a dotted base and
# a from-imported one are two different questions for the pass that records
# what a class derives from.
import a_module_that_declares_a_hierarchy as shapes
from a_module_that_declares_a_hierarchy import Square


class Rounded(shapes.Shape):
    def name(self) -> str:
        return "rounded"


class Sharp(Square):
    def name(self) -> str:
        return "sharp"
