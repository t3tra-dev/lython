# FIXED 2026-08-15, in three layers, each of which was hiding the next. Kept
# because the last one was deferred on a risk that measurement then refuted.
#
#   1. `self._f()` reported "'Holder' inherits builtins.object._f, which Lython
#      does not implement" -- naming a member of object that does not exist.
#      The attribute-call path tried method dispatch, found no method `_f`, and
#      fell into the inherited-object refusal, whose predicate answers for ANY
#      name once the class linearizes onto object. A callable-valued field is
#      now called, not dispatched.
#   2. Under it, `make__lyrt_prim_i64` was collected as the indirect call's
#      target and its result consumed through the object ABI. A primitive-i64
#      clone carries a copy of the original's `callable_type`, so it matched
#      target collection as well as the original did, and the speculation path
#      is keyed on the ORIGINAL's name. A clone is not a callable value.
#   3. Under THAT, the store:
#
#        attribute value '!py.contract<"builtins.function">' is not assignable
#        to field '!py.callable<[], returns = [!py.contract<"builtins.int">]>'
#
#      `isAssignableTo(value->objectValue.contract, fieldType)` compares a
#      RUNTIME REPRESENTATION against a LOGICAL contract. A lowered function
#      reference is `builtins.function` whatever it points at; whether it fits
#      a Callable field is a question about the TARGET, which the bundle names
#      in `functionTarget` and which declares a callable of its own.
#
# ⛔ THE RISK LAYER 3 WAS DEFERRED ON, AND WHY IT WAS NOT ONE. The record said
# the field's runtime lanes come from the Callable contract while the function
# bundle carries its own, so accepting the store needed the two to be
# reconciled first. They never meet: a Callable field is stored BOXED, and the
# boxed path writes the value's payload into the slot's box16 rather than
# splicing a fixed lane tuple into the instance (`storeBoxedFieldPayloadInPlace`,
# Runtime/Ops/AttributeOps.cpp). There is no tuple to disagree about. Measured
# after the relaxation: all four spellings run, all agree with CPython, all net
# zero on the leak gate.
#
# BISECTED, all four now running:
#
#   make() direct .................. 7    (worked throughout)
#   g = make; g() .................. 7    (worked throughout -- a local is fine)
#   self._f = make; self._f() ...... 7
#   self._f = make; g = self._f; g() 7
#   str result ..................... hi
#   int argument ................... 10
#
# Five probes came with it: s8_callable_field_int, _str, _str_module_read,
# _defaultdict_shape and known_field_callable. Found by running
# tests/probe/tools/differential.py over the corpus rather than by picking a
# defect -- the family was five of thirty gaps.
#
# ⛔ STILL REFUSED, and a different family: a field holding a TYPE object
# (`s8_type_object_field`, `known_field_type_object`). `self.t()` is a
# constructor call, the field's contract is `!py.type<...>` rather than a
# Callable, and the ABI reports "callable ABI type has no concrete runtime
# contract". known_field_type_object still shows the layer-1 message for the
# same reason it did here, since the branch added above only covers Callable.
#
# differential: run agrees with CPython now

from typing import Callable


def make() -> int:
    return 7


class Holder:
    def __init__(self, f: Callable[[], int]) -> None:
        self._f: Callable[[], int] = f

    def call(self) -> int:
        return self._f()


print(Holder(make).call())
