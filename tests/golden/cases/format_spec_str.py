# str / bool の format spec を CPython 3.14 の実出力でピン留めする。
s = "abc"
print(f"{s}")
print(f"{s:s}")
print(f"{s:5}|")
print(f"{s:<5}|")
print(f"{s:>5}")
print(f"{s:^5}|")
print(f"{s:5s}|")
print(f"{s:.2}")
print(f"{s:.2s}")
print(f"{s:^8.3s}|")
print(f"{s:*^8}")
print(f"{s:0<6}")
print(f"{s:06}")
print(f"{s:0>4}")
print(f"{s:'^7}")
print(f"{s:.0}|")
print(f"{'':.0}|")
# 非 ASCII の fill / 本文
w = "あいう"
print(f"{w:^7}|")
print(f"{s:あ^7}")
# 幅が長さ以下ならそのまま
print(f"{'hello world':5}")
# bool
print(f"{True}")
print(f"{False}")
print(f"{True:5}|")
print(f"{False:>5}")
print(f"{True:+d}")
print(f"{True:c}" == "\x01")
