# float の shortest round-trip repr が CPython 3.14 と一致することを固定する。
# 期待値は CPython 3.14 の実出力から生成した。
print(0.0)
print(-0.0)
print(1.0)
print(-1.0)
print(0.5)
print(3.14)
print(0.1)
print(1 / 3)
print(2.5)
print(100.0)
print(123456.789)
print(1234567890.12345)

# 固定小数 <-> 指数表記の切替境界 (1e16 / 1e-4)
print(1e15)
print(1e16)
print(9999999999999998.0)
print(0.0001)
print(1e-05)
print(1.5e-05)

# 極値・非正規化数
print(1.7976931348623157e308)
print(8.98846567431158e+307)
print(5e-324)
print(1e-323)
print(4e-323)
print(1e100)
print(1.25e-10)
print(9.109383713928296e-31)
print(6.02214076e23)

# 特殊値 (inf 演算経由; float("inf") 構築は未対応のため)
inf = 1e309
neg_inf = -1e309
print(inf)
print(neg_inf)
print(inf - inf)

# round-trip: リテラルを repr してそのまま読み戻した値と一致する
x = 0.30000000000000004
print(x == 0.1 + 0.2)
print(repr(1.7) == "1.7")
print(str(2.675) == "2.675")
