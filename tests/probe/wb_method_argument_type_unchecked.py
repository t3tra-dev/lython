# FIXED 2026-08-15. A call to a user-defined METHOD or CONSTRUCTOR was not
# checked against the declared parameter types. The body is inlined at the call
# site with the argument VALUE bound to the parameter name, so an argument of
# the wrong type was substituted into the body, and the program then succeeded
# or failed on whatever the body happened to do with it.
#
#     class C:
#         def take(self, xs: list[str]) -> int:
#             return len(xs)
#     C().take({"a": 1, "b": 2})       # compiled, answered len(dict) == 2
#
# A FREE FUNCTION of the same signature has always refused it ("call arguments
# do not match the Callable contract"), so the check existed and the method
# path went round it.
#
# ⭐ HOW IT WAS FOUND, which is the reusable part: not by looking for it. A
# golden written for an unrelated fix printed `False` for
# `Counter({"x": 2, "y": 1}) == Counter()` after three `update` calls, and
# CPython prints True. `Counter.__init__` declares `list[str] | None`; the dict
# went in, `update` iterated its KEYS, and every count came out 1 where CPython
# gives 2 and 1. Below is that program.
#
# ⛔ AND THE PORT ALREADY SAID SO. `collections.py`'s docstring lists, under
# "Deviations from CPython, pending language surface": "Counter()/update()/
# subtract() seed from a list of keys; the mapping/kwargs constructor forms are
# not provided". The sentence was true of the FILE and false of the compiler --
# the form it says is not provided was accepted, and answered wrongly. A
# documented non-feature is only documented if something refuses it.
#
# ⛔ TWO THINGS THE CHECK MUST NOT REFUSE, both found by running it:
#
#   the numeric tower  `def scale(self, x: float)` reached by `scale(3)`.
#                      `isAssignableTo` answers false for int against float --
#                      a free function refuses it and the specializer then
#                      emits a second body -- but an inlined method re-emits
#                      its body at every call site anyway, so the
#                      specialization it would need is the emission about to
#                      happen. Admitted by rung comparison.
#   a synthesized      `@dataclass`/`NamedTuple` give `__eq__` the parameter
#   signature          type Self, and Python's data model gives it `object`:
#                      `TupleA(1) == TupleB(1)` is True in CPython, and
#                      `inherited_post_init_and_cross_class_equality` pins it.
#                      Exempt, because the signature is this compiler's
#                      spelling rather than the program's.
#
# tests: EmitterTest.RejectsMethodArgumentThatViolatesTheDeclaredParameter and
# EmitterTest.DeclaredParameterCheckStillAdmitsTheseThree (emit layer -- the
# repair is a static rejection, so nothing has to run).

from collections import Counter

a: Counter[str] = Counter()
a.update(["x", "x", "y"])
print(a["x"], a["y"], len(a))
print(a.most_common(2))
