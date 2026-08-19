# What this pins: SystemExit caught and read like any other exception, now that
# its argument lands in the payload block instead of the message lane.
#
#     raise SystemExit(4)   # cannot adapt builtins.int to runtime input 3 of
#                           # builtins.SystemExit.__init__
#
# Why this must run: str(e) and e.args are RENDERED from the block at run time,
# and `sys.exit` inside a try has to reach the handler through the same unwind
# every other raise uses -- neither is visible before the program runs.
#
# ⛔ The exit STATUS of an uncaught one is pinned by two error goldens
# (system_exit_int_code, system_exit_empty_message); a case that exits nonzero
# cannot also assert its stdout here.
#
# ⛔ `sys.exit(7)` is only asked whether it was CAUGHT, not what it carries:
# it writes the status into the code slot without also boxing 7 into the
# payload block, so str(e) is "" and e.args is () where CPython gives "7" and
# (7,). Boxing it there needs the 16-word box layout the LOWERING computes, and
# the manifest has no int-shaped store to reach it with. The exit status --
# which is what sys.exit is for -- is right either way.
import sys

try:
    raise SystemExit(4)
except SystemExit as e:
    print("int", e, e.args, len(e.args))

try:
    raise SystemExit("bye")
except SystemExit as e:
    print("str", e, e.args)

try:
    raise SystemExit()
except SystemExit as e:
    print("bare", repr(str(e)), len(e.args))

caught_sys_exit = False
try:
    sys.exit(7)
except SystemExit:
    caught_sys_exit = True
print("sys.exit caught", caught_sys_exit)

try:
    try:
        raise SystemExit(1)
    finally:
        print("finally ran")
except SystemExit:
    print("caught after finally")

caught = 0
try:
    raise SystemExit(2)
except BaseException:
    caught += 1
print("base handler", caught)
