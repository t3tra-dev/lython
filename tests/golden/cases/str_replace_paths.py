# 実行が必要な理由: replace がどの経路を通ったかは型にも診断にも出ない。
# 同じ長さの置換 (丸ごとコピーしてから上書き)、長さの違う置換 (出現数を数えて
# から区間コピー)、空の針 (文字ごとの一般経路)、latin-1 でない入力 (同じく
# 一般経路) が同じ答えを出すことは、値でしか確かめられない。

# 同じ長さ: 1 文字と複数文字。
print("the quick brown fox".replace("o", "0"))
print("abcabc".replace("bc", "XY"))
print("aaaa".replace("aa", "bb"))

# 置換が針をもう一度綴る形。入力ではなく書きかけの出力を探すと壊れる。
print("abab".replace("ab", "ba"))

# 長さが違う: 伸びる / 縮む / 消える。
print("aaa".replace("a", "bb"))
print("aaaa".replace("aa", "b"))
print("aXbXc".replace("X", ""))

# 回数制限。
print("aaa".replace("a", "bb", 2))
print("aaa".replace("a", "b", 0))
print("aaa".replace("a", "b", -1))

# 一致なし: 受け取ったものがそのまま答え。
print("abc".replace("z", "y"))
print("".replace("a", "b"))

# 空の針は文字の間と両端に入る。
print("ab".replace("", "-"))
print("".replace("", "-"))

# latin-1 を越える入力・置換。出力の幅は入るうち最小でなければならない。
print("héllo".replace("l", "L"))
print("abc".replace("b", "✓"))
print("a✓b".replace("✓", "c"))
print("a✓b".replace("✓", "") == "ab")
print("✓✓".replace("✓", "x") == "xx")

# 針が入力より長い。
print("ab".replace("abc", "z"))
