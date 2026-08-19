# What this pins: `raise SystemExit("")` exits 1. It exited 0.
#
# The runner used to read "the message is empty" as "this came from sys.exit,
# use the recorded status", and an empty string is indistinguishable from no
# argument at all under that test -- so a program that meant to fail quietly
# reported success. The argument now goes into the payload block whether or not
# it is a str, so "no argument" and "an empty one" are different shapes, and the
# runner asks the block rather than the message length.
#
# Why this must run: the whole difference is the exit status, which is the one
# thing no compile-time check can see. CPython also writes a blank line to
# stderr here; the exit code is what changed, and what a caller reads.
print("before")
raise SystemExit("")
