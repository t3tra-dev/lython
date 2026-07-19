# str.format (位置/名前/属性参照/ネスト幅/実行時テンプレート) と
# % フォーマットの各変換。CPython 3.14 の実出力でピン。
print("{} {} {}".format(1, 2, 3))
print("{2} {0} {1}".format("a", "b", "c"))
print("{name}={val}".format(name="k", val=9))
print("{0:>{1}}".format("x", 6))
print("{{literal}} {0}".format(7))
print("{0!r:>8}".format("hi"))
print("{:08,d} {:.2f}".format(1234567, 3.14159))
print("{a}{b}{a}".format(a="x", b="y"))
print("value: {v:.3e}".format(v=12345.678))
print("{}".format([1, 2]))
print("{0} {0}".format("dup"))

class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

p = Point(5, -3)
print("{0.x:03d},{0.y:+d}".format(p))
print("x={p.x} y={p.y}".format(p=p))

# 実行時テンプレート (自動番号フィールドのみ、R4: 静的に型付く範囲)
def pick_template(flag: bool) -> str:
    if flag:
        return "[{}] <{:>6}>"
    return "{} / {}"

t = pick_template(True)
print(t.format(42, "ab"))
t2 = pick_template(False)
print(t2.format(1.5, 7))
t3 = "conv {!r} and {!s}"
print(t3.format("q", "w"))

# % フォーマット
print("%d %i %5d %-5d| %05d" % (42, -7, 42, 42, 42))
print("%o %x %X %#o %#x %#X" % (8, 255, 255, 8, 255, 255))
print("%e %E %.2e" % (12345.678, 12345.678, 12345.678))
print("%f %.0f %.10f" % (3.14, 3.14, 3.14))
print("%g %G %.3g" % (0.0001, 1e20, 12345.6))
print("%s %r %a" % ("x", "x", "x"))
print("%c %c" % (65, "B"))
print("%% %5.2f" % 3.14159)
print("%+d % d %+d" % (5, 5, -5))
print("%10.4s|" % "abcdefg")
print("%d %s" % (True, True))
print("%x %#x" % (-255, -255))
print("%e" % 0.0, "%g" % 0.0, "%g" % 1e-5, "%g" % 123456, "%g" % 1234567)
print("%.0e" % 5.0, "%#g" % 1.0, "%#.0f" % 2.0)
print("%s and %s" % (1.5, [1, 2]))
print("%s" % 5)
print("%-8s|%8s|" % ("ab", "cd"))
