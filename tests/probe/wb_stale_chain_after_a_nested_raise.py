# WHAT: an exception raised INSIDE a handler and caught OUTSIDE it leaves the
# inner exception restored as the pending one, so a later uncaught raise prints
# a chain that belongs to an exception already handled.
#
# MEASURED 2026-08-28, and PRE-EXISTING (identical on the binary before the
# `format_exception` chain work):
#
#   lyc  -> the SystemError traceback preceded by ValueError: a and
#           "During handling of the above exception, another exception occurred"
#   3.14 -> the SystemError traceback alone
#
# ROOT: `LyEH_DiscardCurrentException` restores a completing handler's
# __context__ node as the pending exception -- CPython's exception-stack pop --
# and that is right only while the context's own handler is still running. Here
# the raise EXITED the ValueError handler, so by CPython's rules the ValueError
# was popped before the RuntimeError ever reached its handler.
#
# The two neighbours that WORK say where the line is:
#   - a plain caught exception then an uncaught raise: clean.
#   - a nested try inside a handler followed by a bare `raise`: clean, and it
#     is the case the restore exists for.
# So the restore has to be gated on the context's handler still being active,
# which the emitter knows (`exceptHandlerDepth`) and the runtime does not.
try:
    try:
        raise ValueError("a")
    except ValueError:
        raise RuntimeError("b")
except RuntimeError:
    pass
raise SystemError("uncaught")
