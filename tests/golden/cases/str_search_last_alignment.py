# 実行が必要な理由: 前方探索は「窓の 1 つ先」を bloom に問うので、その読みが
# 範囲内かの判定をループの外に出してある — 外れるのは最後の位置ちょうど 1 つだけ
# だから。最後の位置に一致がある形が正しいことは値でしか確かめられない。

s = "the quick brown fox"

# 針が文字列の末尾ぴったりに一致する。
print(s.find("fox"), s.rfind("fox"), s.index("fox"))
print(s.endswith("fox"), s.count("fox"))
print(s.split("fox"), s.partition("fox"), s.rpartition("fox"))
print(s.replace("fox", "cat"))
print(s.replace("fox", "kitten"))

# 針が文字列そのもの。
print("abc".find("abc"), "abc".rfind("abc"), "abc".replace("abc", "x"))
# 針が 1 文字長い。
print("abc".find("abcd"), "abc".replace("abcd", "x"))
# 1 文字の針が末尾に。
print("abc".find("c"), "abc".rfind("c"), "abc".replace("c", "C"))
# 末尾に複数回。
print("aXaXaX".find("aX", 2), "aXaXaX".rfind("aX"), "aXaXaX".count("aX"))
print("aXaXaX".replace("aX", "b"))
# 空文字列と 1 文字。
print("".find("a"), "a".find("a"), "a".replace("a", ""))
# 幅の広い針が末尾に。
print("abc✓".find("✓"), "abc✓".rfind("✓"), "abc✓".replace("✓", "!"))
print("✓✓".find("✓✓"), "✓a✓".replace("✓", "b"))
