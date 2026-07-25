# probe: 4 フィールドのユーザークラスを、別のクラスのフィールド型として使う。
#   段階 4a が全フィールドを box16 スロット化すると、フィールド型としてのクラスの
#   コストは 1 + 自身のフィールド数になる想定なので、これは 5 = 予算ちょうど。
#   4a の前後で通り続けなければならない境界ケース。
#   (`c3de5e7` 時点の probe 集合には「フィールド型として使われるクラス」が
#   Wide(3) / Other(1) / Box(1) しかなく、予算 5 に触る形が 1 件も無かった。)
# axes: width=wNcls(4 fields as a field type) op=construct+read flow=straight budget=5
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 1 2 3 4 / 5 8


class Four:
    def __init__(self, a: int, b: int, c: int, d: int) -> None:
        self.a: int = a
        self.b: int = b
        self.c: int = c
        self.d: int = d


class Holder:
    def __init__(self, v: Four) -> None:
        self.v: Four = v


h = Holder(Four(1, 2, 3, 4))
print(h.v.a, h.v.b, h.v.c, h.v.d)
fresh = Four(5, 6, 7, 8)
h.v = fresh
print(h.v.a, h.v.d)
