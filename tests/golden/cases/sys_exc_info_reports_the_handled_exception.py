# WHAT: `sys.exc_info()` in an `except` block is the exception that block
# handles -- its class, itself, and its traceback -- under every spelling of
# the name, in a method and in a function, and the inner exception while an
# inner handler runs.
#
# WHY THIS IS RUN: which exception is being handled, and the line its
# traceback records, are what the running program raised.
import sys
import sys as system
from sys import exc_info


class Runner:
    def run(self, k: int) -> str:
        try:
            if k == 0:
                raise KeyError("zero")
            return str(10 // k)
        except Exception:
            t, v, tb = sys.exc_info()
            return t.__name__ + ":" + str(v) + ":" + str(tb.tb_lineno)


def f() -> None:
    try:
        raise ValueError("outer")
    except ValueError as e:
        info = exc_info()
        print(info[0] is ValueError, info[1] is e, info[2].tb_frame.f_code.co_name)
        try:
            raise TypeError("inner")
        except TypeError:
            print(system.exc_info()[0].__name__)
        print(sys.exc_info()[0].__name__)


r = Runner()
print(r.run(2), r.run(0))
f()
