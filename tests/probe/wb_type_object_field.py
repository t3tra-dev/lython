# FIXED 2026-08-15, and the record is mostly about the three attempts that
# were reverted first. A field holding a TYPE object was refused:
#
#     callable ABI type has no concrete runtime contract:
#     '!py.type<!py.contract<"Other">>'
#
# ⛔ THE FIRST DIAGNOSIS WAS FALSE and is kept because the shape recurs: it
# read "'Box' inherits builtins.object.t, which Lython does not implement".
# There is no member `t` of object. The attribute-call path tried method
# dispatch, found no method `t`, and fell into the inherited-object refusal,
# whose predicate answers for ANY name once the class linearizes onto object --
# so a field name arrived there looking like a missing dunder.
#
# ⛔ AND THE SECOND WAS THE REPAIR ITSELF. Three layers were built and reverted,
# each moving the refusal one layer down and every one of them building toward
# the same i64 class id:
#
#   1. `runtimeShapeContractName` answering "builtins.type", plus a
#      `@LyType_Shape() -> i64` declaration
#        -> "runtime object header has invalid type 'i64'"
#   2. `primitiveFieldSlot` accepting builtins.type, since a class id is
#      "a whole value in one i64"
#        -> "attribute value has no unbox.i64 primitive for field '_cls'"
#   3. `lowerAttrSet` materialising the class id as a constant
#        -> NO CHANGE: the bundle reaching there is not
#           `RuntimeBundle::Kind::TypeObject`, because it is
#           `Box.__init__`'s PARAMETER and nothing made a TypeObject bundle
#           for an entry argument.
#
# ⭐ AND THE FOURTH LAYER IS WHERE THE PLAN WAS WRONG, not merely unfinished.
# `t(3)` on a parameter holding a runtime class id has to pick a constructor at
# run time, and this compiler has no dynamic dispatch to fall back to -- so
# every layer above was being built to support a thing that would have had to
# be refused at the top. The i64 was never going to arrive anywhere.
#
# ⭐ WHAT IT IS INSTEAD: `type[X]` HAS AN EMPTY PHYSICAL SHAPE. A type object
# carries nothing observable beyond WHICH class it is, and that is in its type,
# so a parameter, a field, a return and a suspension lane all carry it for
# free and constructing through it stays statically resolved. Five sites, each
# one small because the value is not there:
#
#   runtimeValueTypesFor      `!py.type<X>` -> {} (and the refusal below)
#   seedCallableEntry...      a `type[X]` parameter rebuilds its own bundle
#   FunctionTargetCalls       a `type[X]` argument occupies no ABI input
#   lowerAttrSet / AttrGet    the store writes nothing, the read rebuilds
#   emitCall                  a non-Name callee whose type is `type[X]` is
#                             re-spelled as the class name, with the callee
#                             expression still emitted for its effects
#
# ⭐ AND THE GENERATOR SYMPTOM WENT WITH IT, but not through a lane. The
# emitter emits `py.type.object` ONCE per class and every construction shares
# it, so a generator constructing at two or more yields had one live across the
# first suspend. The state machine now SINKS type objects to their uses on the
# clone before liveness runs -- the op is Pure with no operands, so a copy in
# front of each user is the same value -- and nothing is live to need a lane.
# One construction always worked because the value died before the suspend.
#
# ⛔ WHAT IS STILL REFUSED, with a diagnostic that says so: a `type[X]` whose
# class is SUBCLASSED in the program.
#
#     class Base: ...
#     class Derived(Base): ...
#     def make(t: type[Base]) -> Base: return t(3)
#     print(make(Base).n, make(Derived).n)   # CPython 3 30
#
# The empty shape is sound because the type decides the class, and `type[Base]`
# stops deciding it the moment Base has a subclass. ⛔ Why the argument
# specializer does not already cover it (it is the same shape as `f(3)` against
# `def f(x: float)`): a specialized body would have parameter type
# `!py.type<Base>` for the `make(Base)` call, and that spelling is still the
# subclassed one -- `py::TypeType` has no exactness bit, so "exactly Base" and
# "Base or a subclass" are the same type. The mechanism is that bit (or a
# marker on a specialized parameter saying only exact classes reach it), and
# with it the specializer covers both calls.
#
# golden: tests/golden/cases/type_object_representation.py (red-checked)

from typing import Iterator


class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, t: type[Other]) -> None:
        self.t: type[Other] = t


print(Box(Other).t(5).n)


def make(t: type[Other]) -> Other:
    return t(3)


print(make(Other).n)


def gen() -> Iterator[Other]:
    yield Other(1)
    yield Other(2)


for o in gen():
    print(o.n)
