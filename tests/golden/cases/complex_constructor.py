# What this pins: `complex(...)` -- the name, its four arities, and that the
# object it builds is the same one a literal builds.
#
#     print(complex(1, 2))
#     # unresolved name 'complex'
#
# The type was there all along: `1 + 2j` runs, and the manifest carries add /
# sub / mul / truediv / neg / pos / abs / eq / ne / repr / str plus a __new__
# that takes two f64 with defaults. What was missing was the NAME binding and
# the class's own __new__ / __init__ declarations, so the one spelling CPython
# users reach for first was the one that did not work.
#
# Why this must run: the answer is the value, and the constructor's arity is
# resolved against declared overloads -- so `complex(1, 2)` picking the (int,
# int) overload and `complex(0.5)` picking the one-argument one has to end in
# the same repr CPython prints. Arithmetic on a constructed value is here for
# the other half: a literal and a construction must produce the same object, or
# `z + w` would disagree with `1 + 2j` for reasons nothing in the repr shows.
#
# ⛔ Seven __new__ / __init__ pairs rather than one with a union parameter: the
# runtime takes raw f64 and an int argument reaches it through int's unbox.f64,
# which the ABI adapter does per-argument -- but the OVERLOAD is chosen by the
# declared type before any of that, and a union parameter would arrive as a
# union value (tag plus lanes) instead of a number.
#
# ⛔ The empty __init__ is load-bearing, not a placeholder: __new__ builds the
# whole value, but the constructor path calls both, and with no __init__ of its
# own the MRO's next provider is builtins.object's, whose input is boxed --
# "cannot pass concrete object builtins.complex as builtins.object".
z = complex(1, 2)
w = complex(0.5)
print(z, w, complex(), complex(1.0, 2))
print(z + w, z * z, abs(z), -z)
print(z == complex(1, 2), z == 1 + 2j, z != w)
