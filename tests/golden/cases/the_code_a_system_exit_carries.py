# `e.code` of a SystemExit is what CPython computes from `args`: None for no
# argument, the argument for one, the tuple for more. `sys.exit` with a str
# or None (or nothing) is `raise SystemExit(code)`; it took only an int.
# WHY THIS IS RUN: the code is read back from the raised object, and a
# subclass's object answers too.
import sys
for code in (2, "msg", None):
    try:
        sys.exit(code)
    except SystemExit as e:
        print(repr(e.code))
for e in (SystemExit(), SystemExit(1, 2), SystemExit(True)):
    print(repr(e.code))
class Quit(SystemExit):
    pass
try:
    raise Quit(7)
except SystemExit as e:
    print(type(e).__name__, repr(e.code))
