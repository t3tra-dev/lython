# OPEN, and now the ONLY thing between this program and running. Two layers
# above it were removed 2026-08-15 and each was hiding the next:
#
#   1. `self._f()` reported "'Holder' inherits builtins.object._f, which Lython
#      does not implement" -- naming a member of object that does not exist.
#      The attribute-call path tried method dispatch, found no method `_f`, and
#      fell into the inherited-object refusal, whose predicate answers for ANY
#      name once the class linearizes onto object. FIXED: a callable-valued
#      field is called, not dispatched.
#   2. Going through a local (`g = self._f; g()`) got past that and died with
#      "function target 'make__lyrt_prim_i64' returned too few values for
#      result object ABI". A primitive-i64 clone carries a copy of the
#      original's `callable_type`, so indirect target collection matched it,
#      and the speculation path is keyed on the ORIGINAL's name -- nothing
#      downstream recognised the clone as one. FIXED: a clone is not a callable
#      value.
#
# ⭐ WHAT IS LEFT is the STORE, and all four spellings now reach it together:
#
#     attribute value '!py.contract<"builtins.function">' is not assignable to
#     field '!py.callable<[], returns = [!py.contract<"builtins.int">]>'
#
# `isAssignableTo(value->objectValue.contract, fieldTypes[index])` in
# `lowerAttrSet` (Runtime/Ops/AttributeOps.cpp) compares a RUNTIME
# REPRESENTATION against a LOGICAL contract. A lowered function reference is
# `builtins.function` whatever it points at; whether it fits a `Callable[...]`
# field is a question about the TARGET's declared callable, which the bundle
# names in `functionTarget`, not about the representation's name.
#
# ⛔ Why that is not a two-line relaxation: the field's runtime lanes come from
# the Callable contract and the function bundle carries its own (handle plus
# closure values). Accepting the store means the lane shapes have to agree, and
# that is unmeasured -- a check that passes and a store that writes the wrong
# lanes is worse than the refusal.
#
# BISECTED (./build/bin/lyc, all four now identical):
#
#   make() direct .................. 7   (works)
#   g = make; g() .................. 7   (works -- a local callable is fine)
#   self._f = make; self._f() ...... refused at the store
#   self._f = make; g = self._f; g() refused at the store
#   the same with a str result ..... refused at the store
#   the same with an int argument .. refused at the store
#
# So it is the field, and only the field: the callable value itself round-trips
# through a local without complaint.
#
# differential: skip refused; the point is the refusal

from typing import Callable


def make() -> int:
    return 7


class Holder:
    def __init__(self, f: Callable[[], int]) -> None:
        self._f: Callable[[], int] = f

    def call(self) -> int:
        return self._f()


print(Holder(make).call())
