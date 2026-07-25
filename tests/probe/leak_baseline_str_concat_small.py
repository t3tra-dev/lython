# probe: leak -- control: string temporaries created and dropped each iteration (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 3200
# RSS: 30 バイト/回 → リークなし (計測ノイズ ±130 B/回 の範囲)

def once() -> int:
    s = "0123456789abcdef" + "0123456789abcdef"
    return len(s)


total = 0
for _ in range(100):
    total += once()
print(total)
