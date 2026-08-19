# What this pins: `raise SystemExit(3)` exits WITH 3, in silence.
#
# It used to be refused outright ("cannot adapt builtins.int to runtime input 3
# of builtins.SystemExit.__init__"), because an exception's one argument went
# into the message LANE, which is a unicode. The code now rides the exception
# object in a slot of its own, which is where CPython's .code lives too, and the
# top-level runner reads that slot instead of inferring the answer from whether
# the message came out empty.
#
# Why this must run: the exit STATUS is the assertion, and only a process has
# one. The prints show that the raise still unwinds the ordinary way -- finally
# runs, and nothing goes to stderr.
def leave(code: int) -> None:
    try:
        raise SystemExit(code)
    finally:
        print("finally")


print("before")
leave(3)
print("unreachable")
