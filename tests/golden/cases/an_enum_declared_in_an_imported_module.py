# What: an Enum class declared in another file. The desugaring that turns
# `class Color(Enum)` into a plain class with constructed members ran on the
# MAIN module only, so an imported one reached the dialect verifier as
# "'py.class' op unknown base class 'Enum'" -- the compiler's own sentence for
# a class CPython has no trouble with, and the same class in the main module
# has always worked.
#
# Only running it shows the members exist: a member is an INSTANCE held as a
# class attribute, so the names, the values and the identity comparisons are
# what say the desugaring produced objects rather than a compile-time constant.
# The local enum beside the imported one is here because both now share the
# same collected set, and their members must not be confused for each other.
from enum import Enum

import a_module_of_enum_kinds as kinds
from a_module_of_enum_kinds import Color


class Mood(Enum):
    CALM = 10


def describe(c: Color) -> str:
    if c == Color.RED:
        return "hot"
    return "cool"


print("qualified", kinds.Color.RED.name, kinds.Color.RED.value)
print("by name", Color.BLUE.name, Color.BLUE.value)
print("kinds", kinds.Size.LARGE.value, kinds.Tag.A.value, kinds.Step.TWO.value)
print("compared", Color.GREEN == Color.GREEN, Color.GREEN == Color.BLUE)
print("in a function", describe(Color.RED), describe(Color.BLUE))
print("beside a local one", Mood.CALM.name, Color.RED.value + Mood.CALM.value)
