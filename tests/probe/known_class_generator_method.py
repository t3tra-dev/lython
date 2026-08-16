# probe: REPORTED loud: a generator method on a class
# axes: op=generator-method flow=for
# CLASSIFICATION @ 2026-08-17: 3 loud 拒否 (診断)
#   a generator cannot carry a value of contract 'Bag' across a suspension yet
#
# ⛔ RECLASSIFIED TWICE ON 2026-08-17, and the pair is the useful part. It was
# "generator function return annotation is incompatible with inferred Generator
# contract" (the signature computed before the class contracts existed); then
# the single-lane yield refusal, once the yield type was right; and now the
# frame:
#
#   a generator cannot carry a value of contract 'Bag' across a suspension
#
# The `for x in b.xs` loop is rewritten into an index loop now, so the position
# survives -- `known_module_generator_over_field`, the same program with the
# generator at module level, RUNS. What is left is the RECEIVER.
#
# ⭐ AND IT IS NOT "a user class has no lane", which is what the diagnostic
# says. Measured four ways on 2026-08-17:
#
#   class E: (no fields at all)  def each(self): ... yield ... -> REFUSED
#   self.n read into a local BEFORE the first yield ............ REFUSED
#   self.xs likewise ........................................... REFUSED
#   the same body as a module-level function ................... RUNS
#
# An empty class refusing kills the layout reading, and the diagnostic's own
# advice ("read the value into an int local before the first yield") does not
# work, which is worth fixing on its own.
#
# ⭐ WHAT IT ACTUALLY IS: the receiver of a generator METHOD arrives as a
# CLOSURE CAPTURE, not as a positional. The bound form in the frontend IR is
#
#     func.func private @__ly_method$E$each$...$bound$... ()
#       attributes {callable_type = !py.callable<[], returns = ...>,
#                   closure_names = ["self"], closure_types = [!py.contract<"E">]}
#
# -- no positionals at all. `GeneratorStateMachine.cpp` builds `argumentLanes`
# by walking `callable.getPositionalTypes()` (around the `argumentsEligible`
# loop), so the capture gets no lane; the clone's parameter list, built by
# `callableLogicalInputTypes` in ABI/CallableABI.cpp, is positionals ++ kwonly
# ++ vararg ++ kwarg ++ CLOSURES, so the capture IS a clone parameter and
# reaches the "must be builtins.int" check that prints the message above.
#
# `generatorLaneParts` already answers for a source class (a `py.class` in the
# module plus an all-rank-1 layout), so the lane exists -- nothing computes one
# for a capture.
#
# ⛔ BUILDING THE LANE IS NOT ENOUGH, measured 2026-08-17 and reverted. Adding
# the capture to `argumentLanes` (the same loop, over
# `callableClosureTypes(body)`) makes the state machine ACCEPT the bound form:
#
#     [dbg] fn=...$bound$... pos=0 clos=1 argEligible=1 lanes=1
#     [dbg] livesEligible=1
#
# and the identical message then comes from the resume CLONE instead:
#
#     [dbg] clone=...$bound$..._gen_resume inputIndex=4 ctrl=3
#           frameLanes=1 argLanes=1
#
# That walk (ABI/CallableABI.cpp) finds a frame lane with
# `inputIndex - generatorControlCount < frameLanes.size()`, which for input 4
# with a control count of 3 asks for frame lane 1 of 1 and misses. It never
# consults `argumentLanes` at all. So the accounting between the three groups
# -- control, argument, frame -- is what has to be settled first, and guessing
# at it is not something to do in the suspension ABI. The lane change was
# reverted whole: it moves the refusal without changing what any program does.
# CPython 3.14 expects: 6
#
# ⭐ LOCALIZED 2026-08-15, and it is NOT the method, the class, or the loop.
# It is the ITERATOR'S TYPE. Measured, all three in one session:
#
#   for i in range(n): yield i .................... runs  (golden
#                                                   generator_for_loop)
#   i = 0; while i < len(xs): yield xs[i]; i += 1 . runs
#   for x in xs: yield x    (xs: list[int]) ....... refused  <- this shape
#
# The range form's iterator is `!py.contract<"builtins.range_iterator">`, a
# concrete manifest contract with a lane. A list's is
# `!py.protocol<"Iterator", [!py.contract<"builtins.int">]>`, which is
# object-erased: `runtimeContractName` answers "" for it, so the state
# machine's frame-lane scan (`laneEligibleContract`, GeneratorStateMachine.cpp)
# declares the whole generator ineligible and it falls back to the inline
# tier, whose refusal is the message printed today. The iterator is live across
# the yield, which is the only reason it needs a lane at all.
#
# ⛔ Why NOT just name the lane "builtins.object", which is what
# `runtimeShapeContractName` already answers for a protocol and what the rest
# of the ABI does with erased values: the frame lane's contract also types the
# clone's block ARGUMENT (`runtimeContractType(context, lane.contract)`), and
# the continuation's use is `py.next @__next__ : ...(%it : !py.protocol<...>)`.
# An object-typed argument does not spell that operand. The lane would have to
# carry a LOGICAL type beside its physical shape, and the lane grouping (keyed
# by contract name today) would have to key on the logical one. That is one
# mechanism; it was scoped, not built.
#
# ⭐ AND THERE IS A SECOND ONE, FOUND 2026-08-16, which does not touch the lane
# at all: GIVE THE LIST A REAL ITERATOR CONTRACT. The reason `range` works is
# not that ranges are special -- it is that `builtins.range_iterator` is a
# manifest class with `ly.runtime.shape`, `alloc`, `__iter__`, `__next__` and a
# deallocator. `builtins.str_iterator` is the same pattern, and `for c in s:
# yield c` inside a generator gets past this scan for exactly that reason (it
# then fails further downstream, in `str.join`). Those are the only two
# iterator contracts the manifest has:
#
#     ly.runtime.contracts = [..., "builtins.range_iterator",
#                                  "builtins.str_iterator", ...]
#
# A list is iterated as an index walk over the container instead, so `py.iter`
# has nothing concrete to answer with and produces the protocol. A
# `builtins.list_iterator` (header + index + the list handle, mirroring
# str_iterator's header + state + source) would make this generator eligible
# with no change to the lane machinery, and would carry tuple/dict/set behind
# it. Which of the two mechanisms is right is a design call: the lane one is
# general and touches the suspension ABI, this one is local and adds a runtime
# class per container.

from typing import Iterator


class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs

    def each(self) -> Iterator[int]:
        for x in self.xs:
            yield x


b = Bag([1, 2, 3])
total = 0
for v in b.each():
    total += v
print(total)
