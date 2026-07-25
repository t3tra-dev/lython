# probe: 5 フィールドのユーザークラスを別のクラスのフィールド型として使う。
#   段階 4a の box16 スロット化でコストが 1 + 5 = 6 になり予算 5 を超えるため、
#   **4a で「正しい」から「loud 拒否」へ退行する見込みの形**。4a のトレードが
#   probe 集合から見えるようにするために置いた (`c3de5e7` 時点では通る)。
#   退行と誤判定しないこと -- 分類が変わったら、それは 4a が意図した代償が
#   ここに現れたという記録である。
# axes: width=wNcls(5 fields as a field type) op=construct+read flow=straight budget=6
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 1 5 / 6 10


class Five:
    def __init__(self, a: int, b: int, c: int, d: int, e: int) -> None:
        self.a: int = a
        self.b: int = b
        self.c: int = c
        self.d: int = d
        self.e: int = e


class Holder:
    def __init__(self, v: Five) -> None:
        self.v: Five = v


h = Holder(Five(1, 2, 3, 4, 5))
print(h.v.a, h.v.e)
fresh = Five(6, 7, 8, 9, 10)
h.v = fresh
print(h.v.a, h.v.e)
