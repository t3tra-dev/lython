# f-string 全機能: ネスト spec、!r/!s/!a 変換、= デバッグ指定子、
# 式・コンテナ・ユーザークラス __format__。CPython 3.14 の実出力でピン。
x = 42
w = 10
name = "world"
pi = 3.14159
print(f"hello {name}")
print(f"{x} + {x} = {x + x}")
print(f"{x:{w}}|")
print(f"{pi:{w}.3f}")
print(f"{'a':{'>'}{5}}|")
print(f"{x!r}")
print(f"{name!r}")
print(f"{name!s}")
print(f"{name!a}")
print(f"{'héllo'!a}")
print(f"{'あ'!a}")
print(f"{'\U0001f40d'!a}")
print(f"{x=}")
print(f"{x = }")
print(f"{pi=:.2f}")
print(f"{pi=!r}")
print(f"{x!r:>{w}}|")
print(f"")
print(f"a{1}b{2.5}c{'s'}d{True}e{None}f")
print(f"{{escaped}} {x}")
print(f"{[1, 2, 3]}")
print(f"{(1, 'a')}")
print(f"{ {'k': 1} }")

class Angle:
    def __init__(self, degrees: int) -> None:
        self.degrees = degrees

    def __format__(self, spec: str) -> str:
        # ユーザー定義 __format__ は静的ディスパッチで呼ばれる
        return "<" + format(self.degrees, spec) + " deg>"

a = Angle(90)
print(f"{a:04d}")
print(format(a, "+d"))

class Plain:
    def __init__(self) -> None:
        self.v = 1

    def __str__(self) -> str:
        return "plain!"

p = Plain()
print(f"{p}")
print(f"{p!s}")
# format() builtin
print(format(pi, ".3e"))
print(format(x))
print(format(name, "^9") + "|")
print(format(True))
print(format(255, "#x"))
