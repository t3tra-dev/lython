# A module-global `int` is a Python integer, so it grows past the machine
# word exactly as the same name inside a function does. Needs execution: the
# defect printed a wrapped or refused value, not a diagnostic, so only the
# stdout tells the two apart.
#
# The address channel a signal handler reads is `ctypes.c_void_p` and is
# covered by examples/ctypes_signal.py.
counter: int = 1
square: int = 3037000500
total: int = 0


def grow() -> None:
    global counter
    for _ in range(70):
        counter = counter * 2


grow()
print(counter)

square = square * square
print(square)

# Read back after crossing the boundary, and keep going from there.
counter = counter + 1
print(counter)
print(counter % 1000000007)

# A negative one, and the boundary itself.
total = -9223372036854775808
print(total)
total = total - 1
print(total)
total = total * 3
print(total)

# Small values still take the fast path unchanged.
n: int = 7
n = n * 6
print(n)
print(n // 4, n % 4, -n)


def read_it() -> int:
    return counter


print(read_it() == counter)
