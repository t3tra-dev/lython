# フィールドの型が別のユーザークラスであるオブジェクトに対して、入手経路と操作を
# 組み合わせて回す。この表面は 417 テストの時点で実質カバレッジがゼロで、
# 「2 クラスのプログラムが構築するだけで二重解放する」状態が緑のまま通っていた
# (golden/examples のフィールド注釈は int / str / list[...] / dict[...] と
# ネイティブの FileIO しか無かった)。フィールドのスロットはハンドルの再 root で
# 実装されているため、レシーバをどこから得たかで解放の名指し先が変わりうる。
# どの経路・どの操作でも読み出した値が正しく、正常終了しなければならない。
#
# 意図的に含めない形 (いずれも別欠陥として tests/probe/ に記録済み):
#   - if / while / for / try 本体の中での再束縛 (MLIR dominance 失敗)
#   - 借用パラメータ経由の store を呼び出し元が読む形 (書き込みが消える)
#   - 2 段目のフィールド再束縛 o.inner.f = v (同じく消える)


class Inner:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, v: Inner) -> None:
        self.f: Inner = v


def mk() -> Box:
    v: Inner = Inner(0)
    return Box(v)


# --- 入手経路 x 再束縛 ---------------------------------------------------

# インライン構築。
v0: Inner = Inner(0)
b_inline = Box(v0)
fresh0: Inner = Inner(1)
b_inline.f = fresh0
print(b_inline.f.n)

# 関数の戻り値。
b_call = mk()
fresh1: Inner = Inner(2)
b_call.f = fresh1
print(b_call.f.n)


# メソッドの戻り値。
class Factory:
    def make(self) -> Box:
        v: Inner = Inner(0)
        return Box(v)


b_method = Factory().make()
fresh2: Inner = Inner(3)
b_method.f = fresh2
print(b_method.f.n)


# 借用パラメータ (callee の中で観測する)。
def rebind_in_callee(b: Box) -> None:
    fresh: Inner = Inner(4)
    b.f = fresh
    print(b.f.n)


rebind_in_callee(mk())

# コンテナから読み出したレシーバ。
boxes: list[Box] = [mk()]
b_cont = boxes[0]
fresh3: Inner = Inner(5)
b_cont.f = fresh3
print(b_cont.f.n)


# 別オブジェクトのフィールドから読み出したレシーバ。
class Holder:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


h = Holder(mk())
b_field = h.inner
fresh4: Inner = Inner(6)
b_field.f = fresh4
print(b_field.f.n)


# except で束縛した例外のフィールドから読み出したレシーバ。
class Err(Exception):
    def __init__(self, b: Box) -> None:
        super().__init__("boom")
        self.b: Box = b


def boom() -> None:
    raise Err(mk())


try:
    boom()
except Err as e:
    b_exc = e.b
    fresh5: Inner = Inner(7)
    b_exc.f = fresh5
    print(b_exc.f.n)

# メソッドが self のフィールドに store して、呼び出し元が読み戻す。
class SelfStore:
    def __init__(self, v: Inner) -> None:
        self.f: Inner = v

    def bump(self) -> None:
        fresh: Inner = Inner(8)
        self.f = fresh


v1: Inner = Inner(0)
ss = SelfStore(v1)
ss.bump()
print(ss.f.n)

# --- オブジェクト型フィールドを跨いだコンテナの in-place 変異 -------------
# レシーバ自体を別オブジェクトのフィールドから読み出し、その先の list / dict を
# 直接変異させる (別名をローカルに取らない綴り)。


class ListBox:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


class ListHolder:
    def __init__(self, b: ListBox) -> None:
        self.inner: ListBox = b


def mk_listbox() -> ListBox:
    v: list[int] = [10]
    return ListBox(v)


lh = ListHolder(mk_listbox())
lh.inner.f[0] = 99
print(len(lh.inner.f), lh.inner.f[0])


class DictBox:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


class DictHolder:
    def __init__(self, b: DictBox) -> None:
        self.inner: DictBox = b


def mk_dictbox() -> DictBox:
    v: dict[str, int] = {"a": 1}
    return DictBox(v)


dh = DictHolder(mk_dictbox())
dh.inner.f["b"] = 2
print(len(dh.inner.f), dh.inner.f["b"])
del dh.inner.f["a"]
print(len(dh.inner.f))

# --- 4 フィールドのクラスをフィールド型にする (幅の広い合成) --------------


class Four:
    def __init__(self, a: int, b: int, c: int, d: int) -> None:
        self.a: int = a
        self.b: int = b
        self.c: int = c
        self.d: int = d


class FourHolder:
    def __init__(self, v: Four) -> None:
        self.v: Four = v


fh = FourHolder(Four(1, 2, 3, 4))
print(fh.v.a, fh.v.b, fh.v.c, fh.v.d)
fresh6 = Four(5, 6, 7, 8)
fh.v = fresh6
print(fh.v.a, fh.v.d)
