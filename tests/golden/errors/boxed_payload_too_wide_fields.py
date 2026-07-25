# payload box が運べるハンドルは 5 本。それより広く展開されるクラスは、以前は
# box で黙って切り詰められ (読み戻した要素が尾部を失う)、幅ゆえに boxed method
# dispatch からも外れて、存在する `__repr__` が実行時 abort になった。幅が分かる
# のは box の時点なので、そこで拒否する。
#
# 5 つの float フィールド = header 1 + box16 スロット 5 = 6 本。2 int フィールド版
# (cases/boxed_payload_int_fields_fit) が 7 本から 1 本になって通るように
# なったので、予算の境界はこの形に移った。
class Q:
    def __init__(self, a: float, b: float, c: float, d: float,
                 e: float) -> None:
        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.d: float = d
        self.e: float = e

    def __repr__(self) -> str:
        return "Q"


print([Q(1.0, 2.0, 3.0, 4.0, 5.0)])
