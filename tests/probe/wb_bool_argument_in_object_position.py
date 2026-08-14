# OPEN, and NEW (found 2026-08-15). A bool passed where `object` is declared
# fails in lowering; the same parameter accepts an int:
#
#     builtins.bool runtime object header has invalid type 'i1'
#
# BISECTED (./build/bin/lyc against python3.14):
#
#   def f(x: object) -> None: print(x)
#   f(3) ........... 3      (= CPython)
#   f(True) ........ internal error   <- this file
#   print(True) .... True   (= CPython, so bool DOES have a rendering path)
#
# ⭐ ROOT CAUSE: `objectPhysicalHeader` (Runtime/ABI/RuntimeABI.cpp:968) requires
# the value's first lane to be a rank-one i64 memref -- the refcount and
# class-id header every object handle carries. A bool's whole runtime value is
# one i1. It is the only builtin with no header at all: an int that never needed
# a box still materializes one on demand (`materializePrimitiveI64Object`), and
# nothing does that for the truth bit.
#
# ⛔ Why NOT report the unrenderable-value diagnostic the union case gets
# (`describeUnnarrowedOptional`, same function): that message tells the user to
# narrow, and there is nothing here to narrow. `f(True)` is ordinary Python that
# CPython runs; a better message would still be a false rejection.
#
# ⭐ THE REPAIR IS A BOXING PATH FOR THE TRUTH BIT, the twin of
# `materializePrimitiveI64Object`: an object position needs a header, so a bool
# reaching one has to become the singleton it already is in CPython, where
# `f(True)` and `f(False)` hand over the two interned objects. This is NOT the
# numeric tower -- converting to int here would print 1 where CPython prints
# True (see tests/probe/wb_argument_boundary_numeric_tower.py); the header has
# to say bool.
#
# differential: skip internal error; the point is that it never reaches stdout


def f(x: object) -> None:
    print(x)


f(True)
