# probe: フィールド 0 / 1 / 2 個のクラスについて、**別インスタンス同士**の
#   同一性・等価・ハッシュ。object の既定 __eq__ / __hash__ は identity なので
#   CPython はすべて「異なる」と答える。
#   要点は答えではなく **failure shape** の方にある: source class インスタンスの
#   先頭 physical は builtins.object のハンドルと**同じ memref 型**を持つが、
#   同じ意味ではない。entity word (box word 2) を書くのは payload box だけで、
#   素のインスタンスのそれは定数 1 である。したがって「型が合うのだから同じ
#   ハンドルとして渡してよい」と判断する実装は __ly_box_equal / __ly_box_hash に
#   定数 1 を identity として読ませ、**あるクラスの全インスタンスが等しく、
#   同じハッシュになる** — フィールド数に関係なく、compile も run も通り、値だけが
#   静かに間違う。identity を表現の幅や arity から読んではならない
#   (proof kernel の Provenance 規則) の、受け側から見た同じ失敗である。
#   幅の軸を 0 / 1 / 2 で取るのは box 化の要否が physical の個数で変わるため:
#   0 フィールドのクラスは physicals がちょうど 1 個なので "exact match" 経路に
#   落ち、1 個以上とは別の分岐を通る。両方を踏まないと片方だけ直してしまう。
#   kernel/object-methods の実装中に実際に踏んだ (最初の実装は先頭 physical を
#   そのまま alias しており、9 行すべてが True / True / True になった)。
# axes: width=w0cls,w1cls,w2cls op=identity-eq-hash flow=straight
# CLASSIFICATION @ main b580eeb: 3 loud 拒否 (診断)
#   static type !py.contract<"Bare"> does not provide manifest method '__eq__'
# CLASSIFICATION @ kernel/object-methods 225382f: 1 正しい
# CPython 3.14 expects: 4 行 -- False True False / False True False /
#   False True False / False False False


class Bare:
    pass


class One:
    def __init__(self, a: int) -> None:
        self.a = a


class Two:
    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b


b1 = Bare()
b2 = Bare()
o1 = One(1)
o2 = One(1)
t1 = Two(1, 2)
t2 = Two(1, 2)
print(b1 == b2, b1 != b2, b1 is b2)
print(o1 == o2, o1 != o2, o1 is o2)
print(t1 == t2, t1 != t2, t1 is t2)
print(hash(b1) == hash(b2), hash(o1) == hash(o2), hash(t1) == hash(t2))
