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
# A str is two lanes either way, so the count is not what changed -- the
# `continue` edge is, and it makes the yielded value a merge whose lanes the
# resume path counts differently. Reading the message as "multi-lane yields are
# unsupported" is what this note exists to prevent: they are supported, and
# every line-filtering generator is written with the skip.
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
