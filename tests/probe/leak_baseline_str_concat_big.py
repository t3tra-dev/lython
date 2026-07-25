# probe: leak -- control: string temporaries created and dropped each iteration (40000 iterations)
# axes: op=leak-loop iterations=40000
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 1280000
# RSS: -22 バイト/回 → リークなし (計測ノイズ ±130 B/回 の範囲)

def once() -> int:
    s = "0123456789abcdef" + "0123456789abcdef"
    return len(s)


total = 0
for _ in range(40000):
    total += once()
print(total)
