# A user exception held as an ordinary value (never raised) still answers its
# fields. The field block hangs off extended header word[4], so it is reached
# through primitives that take the header and hand back a raw block word --
# a derivation the release planner could not see. The exception's last direct
# use was the block lookup, so its release landed BEFORE the box words were
# read and before the payload was retained: the read dereferenced freed
# storage and the process died with SIGSEGV.
class Where(ValueError):
    def __init__(self, msg: str, pos: int) -> None:
        self.msg = msg
        self.pos = pos
        self.lineno = pos + 1
        super().__init__(msg)


# The field read is the value's last use: nothing keeps the exception alive
# past the block lookup.
w = Where("bad", 4)
print(w.pos)

w2 = Where("bad", 4)
print(w2.msg)

w3 = Where("bad", 4)
print(w3.lineno)


# A field read straight off a temporary, with no local at all.
print(Where("temp", 7).pos)
print(Where("temp", 7).msg)


# Several reads of the same value: only the last one ends its lifetime.
w4 = Where("multi", 2)
print(w4.msg, w4.pos, w4.lineno)


# The exception returned from a function and consumed at the call site.
def make(pos: int) -> Where:
    return Where("made", pos)


print(make(9).pos)
m = make(3)
print(m.msg, m.lineno)


# A field value escaping the exception outlives it: the read takes a
# reference, so the string is still valid after the owner is gone.
def message(pos: int) -> str:
    return Where("escaped", pos).msg


got = message(1)
print(got)
print(len(got))


# A field rebind on a plain value, then a read that ends the lifetime.
w5 = Where("first", 0)
w5.msg = "second"
print(w5.msg)

# str(e) still works alongside the field block.
w6 = Where("shown", 5)
print(str(w6))
w7 = Where("shown", 5)
print(str(w7), w7.pos)
