# オブジェクト型フィールドの再束縛を、現在サポートされている制御フローの各形で
# 回す。フィールドのスロットはハンドルの再 root で実装されているため、再束縛と
# 読み出しの間にどの制御構造が挟まるかで解放の名指し先が変わりうる。例外経路と
# 早期 return を跨いでも読み出した値が正しく、正常終了しなければならない。
#
# 意図的に含めない形 (最早境界での診断になっておらず MLIR dominance 失敗になる。
# tests/probe/flow_{ifone,ifboth,nestedif,while,for,trybody}_w1obj.py に記録):
#   if の片アーム / 両アーム / ネスト if / while / for / try 本体の中での再束縛。
# これらが診断化または解消したら、この golden に移してよい。


class Inner:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, v: Inner) -> None:
        self.f: Inner = v


def mk() -> Box:
    v: Inner = Inner(0)
    return Box(v)


# 直線。
b0 = mk()
a0: Inner = Inner(1)
b0.f = a0
print(b0.f.n)

# except ハンドラの中での再束縛。
b1 = mk()
try:
    raise ValueError("x")
except ValueError:
    a1: Inner = Inner(2)
    b1.f = a1
print(b1.f.n)

# finally の中での再束縛。
b2 = mk()
try:
    pass
finally:
    a2: Inner = Inner(3)
    b2.f = a2
print(b2.f.n)


# 早期 return を跨ぐ再束縛。
def early(flag: bool) -> int:
    b = mk()
    x: Inner = Inner(4)
    b.f = x
    if flag:
        return b.f.n
    y: Inner = Inner(5)
    b.f = y
    return b.f.n


print(early(True))
print(early(False))


# 複数 return を跨ぐ再束縛。
def multi(k: int) -> int:
    b = mk()
    x: Inner = Inner(6)
    b.f = x
    if k == 0:
        return b.f.n
    if k == 1:
        y: Inner = Inner(7)
        b.f = y
        return b.f.n
    z: Inner = Inner(8)
    b.f = z
    return b.f.n


print(multi(0))
print(multi(1))
print(multi(2))


# 例外が再束縛を跨いで伝播し、ハンドラ側で読み戻す。
class Err(Exception):
    def __init__(self, b: Box) -> None:
        super().__init__("boom")
        self.b: Box = b


def raise_after_rebind() -> None:
    b = mk()
    x: Inner = Inner(9)
    b.f = x
    raise Err(b)


try:
    raise_after_rebind()
except Err as e:
    print(e.b.f.n)
