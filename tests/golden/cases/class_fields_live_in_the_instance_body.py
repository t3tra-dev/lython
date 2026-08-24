# 実行が必要な理由: フィールドがインスタンスの body に移ったことは型でも診断でも
# 観測できない。コンテナに入れて読み戻した値が正しいことでしか確かめられない。
#
# 6 本に展開されていた形 (header + float 4 本 + str の box 1 本) を、コンテナに
# 入れて読み戻す。box のレーンは 3 本しかないので、フィールドがレーンを取って
# いた頃はこのプログラムはコンパイル自体が拒否されていた。
class Point:
    def __init__(self, x: float, y: float, z: float, w: float, tag: str) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.w: float = w
        self.tag: str = tag


points = [Point(1.0, 2.0, 3.0, 4.0, "a"), Point(5.0, 6.0, 7.0, 8.0, "b")]
for p in points:
    print(p.x, p.y, p.z, p.w, p.tag)

# body の書き込みもコンテナ越しに観測できる: 読み戻したインスタンスへの store は
# 同じ block に落ちるので、次に読んだときに見える。
points[0].tag = "c"
points[0].x = 9.0
print(points[0].x, points[0].tag)

# int フィールドは body の 1 語。以前は header の空き語に置かれていて 8 本で
# 尽きていたので、9 本目からは別の storage に落ちていた。
class Wide:
    def __init__(self) -> None:
        self.a: int = 1
        self.b: int = 2
        self.c: int = 3
        self.d: int = 4
        self.e: int = 5
        self.f: int = 6
        self.g: int = 7
        self.h: int = 8
        self.i: int = 9
        self.j: int = 10


w = Wide()
print(w.a, w.e, w.i, w.j)
w.j = 99
print(w.i, w.j)
