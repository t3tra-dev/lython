# probe: 4 フィールドのユーザークラスを、別のクラスのフィールド型として使う。
#   4a の前後で通り続けなければならない境界ケースとして置いた
#   (`c3de5e7` 時点の probe 集合には「フィールド型として使われるクラス」が
#   Wide(3) / Other(1) / Box(1) しかなく、予算 5 に触る形が 1 件も無かった)。
#   **この probe は役目を果たした**: 4a で box-fronting が入った直後に loud に
#   なり、それが早期シグナルとして機能して「int フィールドの contract 形
#   placeholder が 1 個あたり 3 ハンドル使っていた」ことが判明した
#   (`Four` は 5 ではなく 13 ハンドルに展開されていた)。placeholder 削除後は
#   `kernel/4a` (`a36d881`) で**通る** — 実測 3/3 + libgmalloc。
# axes: width=wNcls(4 fields as a field type) op=construct+read flow=straight budget=1
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
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
