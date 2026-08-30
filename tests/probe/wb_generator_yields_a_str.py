# A generator that yields a STR from a loop with a `continue` is refused:
#
#   source generator next lowering currently supports yields whose runtime
#   value is a single lane, and '!py.contract<"builtins.str">' has 2
#
# ⛔ THE MESSAGE NAMES THE LANE COUNT AND THE TRIGGER IS THE `continue`. The
# same generator without it compiles and runs:
#
#     def lines(text: str):
#         for line in text.splitlines():
#             yield line            # fine
#
#     def lines(text: str):
#         for line in text.splitlines():
#             if not line:
#                 continue
#             yield line            # refused
#
# A str is two lanes either way, so the count is not what changed, and the
# message is not about the yield at all: it is the FALLBACK path's complaint.
#
# ⭐ THE CAUSE, read off the state machine's eligibility scan by printing what
# it declines (`GeneratorStateMachine.cpp`, the `laneEligibleContract` call over
# `liveAfterYield`):
#
#     [gen-eligible] live value has no lane: i1
#
# The `continue` makes a raw `i1` -- the loop's own condition flag -- LIVE
# across the suspension, and a generator frame lane is keyed on a runtime
# CONTRACT. A bare i1 has none, so the whole generator is declined and falls to
# the single-lane tier, which then refuses the str for a reason that has
# nothing to do with why it got there.
#
# ⛔ SO THE REPAIR IS A PRIMITIVE FRAME LANE, not a wider yield. The frame
# already carries i1s -- every control lane is an (i64, i1) pair -- but
# `GeneratorResumeLane` is built from a contract name and its physical parts,
# so a scalar lane is a new kind threaded through the frame layout, the save
# and the claim helpers. That is the size of it, which is why it is recorded
# rather than attempted here.
#
# ⭐ THE DIAGNOSTIC WAS ITS OWN DEFECT AND IS FIXED: the refusal now carries
# the state machine's reason, so the message names the i1 rather than the lane
# count alone. Pinned by `DriverTest.ARefusedGeneratorNamesWhatSentItDown`.
#
# ⭐ AN EARLIER DRAFT OF THIS PROBE CLAIMED THE PLAIN SHAPE FAILED, and it does
# not. The claim was written from the failing PROGRAM rather than from a
# reduction, and the reduction says something else. Run the probe before
# writing what it proves.
def lines(text: str):
    for line in text.splitlines():
        if not line:
            continue
        yield line


for line in lines("a\n\nb\n"):
    print(line)
