# OPEN, and the largest remaining cluster in the differential: SEVEN probes,
# one cause. Measured 2026-08-15 by running each and reading the first line.
#
#   rebind_gen_w1dict   rebind_gen_w1obj   rebind_gen_w1str
#   rebind_gen_w2float  rebind_gen_w3int   rebind_gen_w3list
#   rebind_gen_wNcls
#
# Every one of them:
#
#     source generator next lowering currently supports int yields
#
# They differ only in what the generator yields -- a dict, an object, a str, a
# float, a list, a user class -- and rebind_gen_w3int is in the set because its
# yield is a Box holding an int, not an int. So the seven are not seven shapes
# of one defect; they are one defect seen through seven payload types, and any
# repair closes all of them at once -- though see below: one of the seven has a
# second gate behind the first, so "one repair" means widening more than one.
#
# ⭐ THE GATE IS ONE LINE, and as of 2026-08-15 it states the real requirement:
# `lowerSourceGeneratorNext` (Runtime/Ops/SourceGenerator.cpp) asks whether the
# element's runtime value is a SINGLE LANE, which is what the code below needs,
# rather than whether the contract is spelled "builtins.int". Measured: int
# yields still run, and every probe here still refuses -- a Box's runtime value
# is its class layout, not a lone handle, so none of them had the property the
# old spelling was standing in for. Zero programs fixed by that change; it
# renamed the requirement and moved one probe to its real next blocker.
#
# ⭐ AND rebind_gen_w3int IS BEHIND A SECOND GATE. With the first one stating
# lanes, it gets past and reports:
#
#     source generator next lowering currently supports only straight-line
#     pure int yield bodies
#
# Still a payload restriction, and still spelled as a contract name, but a
# different one further in -- so the cluster needs BOTH gates widened, and
# whoever takes it should expect a third.
#
# ⭐ AND THE REASON IT IS INT-ONLY IS STRUCTURAL, not a missing case in a switch.
# Everything below that gate is written around a single SSA value per yield:
#
#     struct SourceYieldPlan { mlir::Value value; ... };
#
# An int's runtime value at that point is one i64 lane, so one value is the whole
# payload. A str is two lanes, an object is a handle plus whatever its layout
# carries, and a union is a tag plus every member's lanes. Generalising means the
# plan carries a lane GROUP and the suspended state stores one, which is the same
# widening the generator frame would need -- so this is a mechanism to add, not a
# branch to fix, and it is worth scoping as one item rather than seven.
#
# ⛔ Not the same thing as the generator defects already recorded
# (wb_generator_resume_raise_unwind, the frame's unwind edge): those are about
# what happens when a generator RAISES, and they reproduce with int yields.
#
# differential: skip refused; the point is the refusal

from typing import Iterator


class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def gen() -> Iterator[Box]:
    yield Box("abcd")


for o in gen():
    print(len(o.f), o.f)
