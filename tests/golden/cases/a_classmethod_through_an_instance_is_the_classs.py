# WHAT: a classmethod called through an instance is the class's call, as
# CPython binds it to the instance's type: `b"x".fromhex(...)` is bytes',
# `bytearray(...).fromhex(...)` answers a bytearray, `{}.fromkeys(ks)` and
# `"s".maketrans(x, y)` are dict's and str's. The receiver is still evaluated,
# before the arguments; and a name bound to one inside a loop is typed for the
# code after it, as the class spelling is.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the repair replaces a
# refusal, so what has to be shown is the VALUE each spelling computes and
# the order the receiver and the arguments ran in, which only execution shows.

def make() -> bytes:
    print("receiver")
    return b"zz"
def arg() -> str:
    print("argument")
    return "4142"
print(make().fromhex(arg()))
d: dict[str, int] = {"a": 1}
e = d.fromkeys(["x", "y"], 7)
print(e, d)
class Holder:
    def __init__(self) -> None:
        self.b = bytearray(b"q")
        self.s = "abc"
h = Holder()
print(h.b.fromhex("ff"), h.s.maketrans("ab", "cd"))
for i in range(300):
    t = b"abc".fromhex("00ff")
    u = {i: i}.fromkeys([i])
print(t, u)
class Mine:
    def fromhex(self, s: str) -> str:
        return "mine " + s
print(Mine().fromhex("x"))
for i in range(3):
    loop_bytes = b"a".fromhex("00ff")
    loop_keys = {i: i}.fromkeys([i])
    loop_class = dict.fromkeys([i])
    loop_table = "q".maketrans("a", "b")
print(loop_bytes, loop_keys, loop_class, loop_table)
print(b"".fromhex("") == bytes.fromhex(""), isinstance(bytearray().fromhex(""), bytearray))
