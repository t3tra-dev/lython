# フィールドの型が別のユーザークラスである合成オブジェクトを構築して読み出す。
# `self.inner = b` はスロットのハンドルを再 root するので、そのインスタンス
# (メソッド内の self = 生成マーカーを持たない) の解放が代入前のハンドルを名指す
# 状態で代入側も同じハンドルを解放すると、構築時点で二重解放になる
# (`Ly_DecRef observed non-positive refcount` で abort、出力は 1 行も出ない)。
# 読み出した値が毎回正しく、正常終了しなければならない。


class Inner:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Holder:
    def __init__(self, b: Inner) -> None:
        self.inner: Inner = b


# 生成のみ (読み出しなし) でも構築時に落ちていた。
h0 = Holder(Inner(1))
print("built")

# ネストした読み出しと、ローカルに取り出してからの読み出し。
h1 = Holder(Inner(2))
print(h1.inner.n)
taken = h1.inner
print(taken.n)

# フィールドが 2 つ、どちらもユーザークラス。
class Pair:
    def __init__(self, a: Inner, b: Inner) -> None:
        self.a: Inner = a
        self.b: Inner = b


p = Pair(Inner(3), Inner(4))
print(p.a.n, p.b.n)

# 3 階層。中間のインスタンスも呼び出し由来ではなくインラインで作る。
class Mid:
    def __init__(self, i: Inner) -> None:
        self.i: Inner = i


class Top:
    def __init__(self, m: Mid) -> None:
        self.m: Mid = m


t = Top(Mid(Inner(5)))
print(t.m.i.n)

# 内側の型が str フィールドを持つ場合 (payload が box-fronted になる組み合わせ)。
class Named:
    def __init__(self, s: str) -> None:
        self.s: str = s


class NamedHolder:
    def __init__(self, v: Named) -> None:
        self.v: Named = v


nh = NamedHolder(Named("ab"))
print(nh.v.s)

# 関数の戻り値から受け取った Holder でも同じ。
def mkh() -> Holder:
    return Holder(Inner(6))


print(mkh().inner.n)

# 関数スコープの中で構築・読み出し。
def run() -> int:
    local = Holder(Inner(7))
    return local.inner.n


print(run())
