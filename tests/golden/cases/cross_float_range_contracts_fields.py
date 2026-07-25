# Cross-track: kernel/contract-audit backed float.__floordiv__ / __mod__ /
# __round__ and range's Sequence protocol (__len__ / __getitem__ /
# __contains__), and made `1.0 / 0.0` raise instead of yielding inf. Its own
# goldens all use module-level locals. kernel/4a changed where a field's value
# is stored and how it is read back, so every one of those methods has to work
# when its receiver is loaded out of a field slot or a container element, and
# the ZeroDivisionError has to be raised from there too.
#
# The sign cases carry the weight, as in float_floordiv_mod_round: -7.5 // 2.0
# and 7.5 % -2.0 are where a separately-derived floordiv/mod pair diverges.


class Nums:
    def __init__(self, a: float, b: float) -> None:
        self.a: float = a
        self.b: float = b

    def fdiv(self) -> float:
        return self.a // self.b

    def mod(self) -> float:
        return self.a % self.b


n = Nums(-7.5, 2.0)
print(n.fdiv(), n.mod())
print(n.a // n.b, n.a % n.b)
m = Nums(7.5, -2.0)
print(m.a // m.b, m.a % m.b)
print(round(n.a), round(n.a, 1), round(m.b))

fs: list[float] = [-7.5, 2.0]
print(fs[0] // fs[1], fs[0] % fs[1], round(fs[0]), round(fs[0], 1))

fd: dict[str, float] = {"a": -7.5, "b": 2.0}
print(fd["a"] // fd["b"], fd["a"] % fd["b"])


class Span:
    def __init__(self, r: range) -> None:
        self.r: range = r


s = Span(range(0, 10, 3))
print(len(s.r), s.r[0], s.r[2], 6 in s.r, 5 in s.r)
total = 0
for v in s.r:
    total = total + v
print(total)

rs: list[range] = [range(4), range(1, 5)]
print(len(rs[0]), rs[0][3], 3 in rs[0], len(rs[1]), rs[1][0])


def blow(x: Nums) -> float:
    return x.a / x.b


z = Nums(1.0, 0.0)
try:
    print(blow(z))
except ZeroDivisionError as e:
    print("truediv", e)
try:
    print(z.a // z.b)
except ZeroDivisionError as e:
    print("floordiv", e)
try:
    print(z.a % z.b)
except ZeroDivisionError as e:
    print("mod", e)

zs: list[float] = [1.0, 0.0]
try:
    print(zs[0] / zs[1])
except ZeroDivisionError as e:
    print("container", e)
