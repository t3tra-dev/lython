# payload box が運べるハンドルは 3 本。それより広く展開されるクラスは、以前は
# box で黙って切り詰められ (読み戻した要素が尾部を失う)、幅ゆえに boxed method
# dispatch からも外れて、存在する `__repr__` が実行時 abort になった。幅が分かる
# のは box の時点なので、そこで拒否する。
#
# 3 つの float フィールド = header 1 + float 3 = 4 本で、これが境界のすぐ外側。
# 2 フィールド版は通る。予算は lane 数そのものであり、box が 16 語 (lane 5 本)
# から 12 語 (lane 3 本) に狭まったときに 5 本から 3 本へ動いた。
class Q:
    def __init__(self, a: float, b: float, c: float) -> None:
        self.a: float = a
        self.b: float = b
        self.c: float = c

    def __repr__(self) -> str:
        return "Q"


print([Q(1.0, 2.0, 3.0)])
