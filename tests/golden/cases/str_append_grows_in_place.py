# 実行が必要な理由: `s += x` がブロックをその場で伸ばしたかどうかは型にも診断にも
# 出ない。書き換えてはいけない参照を書き換えていないことは、読み戻した値でしか
# 確かめられない。
#
# CPython の `unicode_concatenate` と同じ条件で伸ばす: フレームが自分の参照を
# 手放す形 (`s += x` の左辺) で、かつ他に誰も持っていないとき。


def accumulate(n: int) -> str:
    s = ""
    for i in range(n):
        s += "ab"
    return s


print(len(accumulate(1000)))
print(accumulate(3))

# 別名が残っているときは伸ばせない。
s = "xy"
t = s
s += "z"
print(s, t)

# 借りた参照 (パラメータ) も伸ばせない: 呼び出し元がまだ持っている。
# accumulate が返す文字列は追記のために余裕を持っているので、ここが
# その場追記に落ちると呼び出し元の値が変わる。
def extend(p: str) -> str:
    p += "!"
    return p


base = accumulate(4)
grown = extend(base)
print(base, grown)

# 幅が上がる追記はその場ではできない (latin-1 のブロックに UCS-2 は入らない)。
w = "ab"
w += "✓"
print(w, len(w))

# 幅が下がる方向は入る。
v = "✓"
v += "ab"
print(v, len(v))

# 自分自身への追記。
u = "ab"
u += u
print(u)
