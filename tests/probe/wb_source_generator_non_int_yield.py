# FIXED 2026-08-15, and the record is worth more than the repair: the cluster
# was seven probes, and the cause named in this file for two sessions was WRONG.
#
#   rebind_gen_w1dict   rebind_gen_w1obj   rebind_gen_w1str
#   rebind_gen_w2float  rebind_gen_w3int   rebind_gen_w3list
#   rebind_gen_wNcls
#
# All eight (these seven and this file) now agree with CPython 3.14.
#
# ⛔ THE RECORDED CAUSE WAS FALSE. It read: "everything below the gate is
# written around `struct SourceYieldPlan { mlir::Value value; }`, so a str is
# two lanes and generalising means the plan carries a lane GROUP -- a mechanism
# to add, not a branch to fix." Every sentence about `SourceYieldPlan` is true
# and none of it was the reason, because the failing programs never needed that
# code to work. Refuted in one command:
#
#     def gen() -> Iterator[str]: yield "abcd"     -> runs, prints "4 abcd"
#
# A str is TWO LANES and it already worked, through
# `emitStateMachineGeneratorResume`, which has carried lane groups all along.
# This is [[lython-localization-discipline]] exactly: a cause that explains the
# failing program was never checked against a program that WORKS.
#
# ⭐ THE REAL CAUSE, one predicate. `laneEligibleContract` in the state
# machine's eligibility scan (GeneratorStateMachine.cpp) asked
# `manifest.valueShape(contract)`, and a SOURCE class has no manifest shape --
# its layout is computed from its ClassOp by `runtimeValueTypesFor`. So every
# generator yielding a user class was declared ineligible and fell back to the
# int-only inline tier, whose refusal is the message this file used to carry.
# `builtins.str` has a manifest shape, which is the whole reason a str yield ran
# and a Box yield did not.
#
# ⭐ AND TWO MORE DEFECTS WERE BEHIND IT, both invisible until the lane opened,
# both of them wrong ANSWERS rather than refusals:
#
#   1. Release placement. `insertImmediateSuccessorReleases` pinned liveness on
#      the group's physical uses only. A field read loads the instance's box
#      words and assembles a BORROWED memref from them, so the loads are the
#      last physical use while the borrow is retained several ops later -- the
#      deallocator landed between the two and `Ly_IncRef` resurrected freed
#      storage. `findReleaseInsertion` and `releaseOwnedGroupByLiveness` both
#      already called `collectBoxWordDerivedViews`; this one did not.
#      Symptom: `for o in gen(): print(o.f)` printed an EMPTY LINE.
#   2. The borrowed-entry return retain. `insertBorrowedReturnRetains`
#      re-derived each result's operand offset by accumulating deallocator
#      widths, which is wrong for a resume clone (its int results are raw
#      (i64, i1) pairs, one memref of release interface). The wrong offset fell
#      through to the contract-blind deallocator lookup, which is ambiguous as
#      soon as two source classes share a lane shape -- so adding an UNRELATED
#      second class moved the retain from the instance header onto its field
#      box. Now it reads the declared `ly.ownership.owned_results`.
#
# ⭐ WHAT IS STILL REFUSED, and both are LIVE-VALUE lane questions rather than
# yield questions. A generator that iterates a list (`for x in xs: yield x`)
# keeps a protocol-typed iterator live across the yield and protocols have no
# lane -- localized in tests/probe/known_class_generator_method.py, where the
# range and while spellings that RUN are recorded beside it. And a generator
# that CONSTRUCTS instances at two or more yields keeps the shared
# `py.type.object` live across the first one; that is the type-object item
# (tests/probe/wb_type_object_field.py): `py.type.object` is emitted once and
# used by every construction, so with two constructions it is live across a
# yield, and `type[X]` has no lane. One construction is fine because the value
# dies before the suspend.
#
# golden: tests/golden/cases/generator_instance_yield.py (red-checked; also in
# LYTHON_LEAK_GATE_CASES, since two of the three repairs are reference counts)

from typing import Iterator


class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def gen() -> Iterator[Box]:
    yield Box("abcd")


for o in gen():
    print(len(o.f), o.f)
