# A `str` and a `bytes` literal built one million times inside one frame.
#
# What this pins: the buffer feeding `builtins.str.__new__` /
# `builtins.bytes.__new__` must not live on the frame. It used to be a
# `memref.alloca` plus one store per byte, and an alloca outside the entry block
# is not `AllocaInst::isStaticAlloca()`, so LLVM lowers it to a runtime stack
# adjustment that nothing reclaims until the function returns. The frame grew by
# the length of the literal on every iteration and the stack guard raised
# RecursionError -- measured at 275,000 iterations of a 20-byte literal.
#
# Why one million and not a lowered `ulimit`: the count has to be large enough
# that no plausible host stack limit lets the old lowering through. 1,000,000
# iterations of a 20-byte literal is ~32 MB of frame growth at the old lowering,
# against 8 MB of default stack here and well past a generous one, so the case
# fails on the defect rather than on how the host is configured. It also does not
# need a subprocess or a resource limit to be set, and it costs ~1.5 s because the
# loop body is cheap once the buffer is a shared read-only global.
#
# The `int` arm is the control: the identical loop shape without a literal in it
# survived 4,000,000 iterations on the defective build, so a failure here is the
# literal and not the loop.
def literal_loop(n: int) -> int:
    total: int = 0
    i: int = 0
    while i < n:
        s: str = "hello, literal world"
        total = total + len(s)
        i = i + 1
    return total


def bytes_literal_loop(n: int) -> int:
    total: int = 0
    i: int = 0
    while i < n:
        b: bytes = b"hello, literal bytes"
        total = total + len(b)
        i = i + 1
    return total


def control_loop(n: int) -> int:
    total: int = 0
    i: int = 0
    while i < n:
        k: int = 20
        total = total + k
        i = i + 1
    return total


print(literal_loop(1000000))
print(bytes_literal_loop(1000000))
print(control_loop(1000000))

# Two occurrences of one literal now share a read-only global, so the values a
# literal produces must still compare and print as themselves. (`is` is not
# spelled here: Lython rejects identity on value types by design.)
a: str = "shared literal"
b2: str = "shared literal"
print(a == b2, len(a), a)
print(a.upper(), a + "!", a[7:], b"shared" + b"!")
