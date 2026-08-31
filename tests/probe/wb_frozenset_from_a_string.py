# `frozenset(x)` binds its element parameter from the ARGUMENT'S OWN parameter,
# so it works for the arguments that have one and not for the arguments whose
# element type has to be read from how they iterate. Measured:
#
#     frozenset([1, 2])     ok      list[T] has T
#     frozenset({1, 2})     ok      set[T] has T
#     frozenset("ab")       refused str has no element parameter
#     frozenset(range(3))   refused range has none either
#     frozenset((1, 2))     refused tuple's parameters are positional
#
# The refusal is "class instantiation leaves unbound static type parameters",
# and `set("ab")`, `list("ab")` and `tuple("ab")` all work -- because those
# three go through the constructor DESUGAR
# (`tryEmitContainerConstructorCall`, EmitterIterators.cpp), whose ctor list is
# list/set/tuple/dict. frozenset is not in it and reaches the manifest
# signature instead.
#
# ⭐ THE GENERAL REPAIR IS THE ITERATION ELEMENT: an `Iterable[T]` parameter
# should bind T to what the argument YIELDS, which `TypeSystem::
# iterationElementType` already answers for all five. A user function annotated
# `Iterable[str]` accepts a str today, so the assignability side is fine; what
# is missing is the DEDUCTION at a class instantiation.
#
# ⛔ Adding frozenset to the desugar is not the same fix: the comprehension
# that desugar builds produces a `builtins.set`, and there is no literal
# spelling for a frozenset to produce instead.
print(sorted(frozenset("ab")))
