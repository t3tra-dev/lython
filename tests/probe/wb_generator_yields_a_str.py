# FIXED 2026-08-30. `if not line: continue` before a yield leaves an
# `arith.constant true` live across the suspension, and a frame lane is keyed
# on a runtime CONTRACT, so a bare `i1` had nowhere to live: the state machine
# declined the whole generator and the tier below refused it for its own limit
# ("yields whose runtime value is a single lane", of a `str`) -- which is why
# this read as a str-yield defect and was not one. `len(line) == 0` in the same
# place compiles, and that is the whole difference.
#
# ⭐ THE REPAIR IS THE ONE THE TYPE OBJECTS ALREADY GOT: a value with no
# runtime representation is REMATERIALIZED, not threaded. The sink in
# GeneratorStateMachine.cpp now takes every operand-free pure single-result op
# rather than naming `py.type.object`, because that is exactly the class of
# values a copy in front of each user reproduces.
#
# ⛔ NOT a scalar frame lane, which is what the earlier note scoped: a lane
# also types the clone's block argument and is what the continuation reads the
# value back from, so it would still need this rematerialization at the far
# end.
#
# Golden: cases/a_generator_skips_the_empty_lines.
def lines(text: str):
    for line in text.splitlines():
        if not line:
            continue
        yield line


for line in lines("a\n\nb\n"):
    print(line)
