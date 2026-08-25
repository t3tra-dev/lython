# 実行が必要な理由: ASCII 高速路を通ったか UCD テーブルを引いたかは型にも診断にも
# 出ない。両方が同じ答えを出すこと、そして境界 (latin-1 の上半分、幅が上がる写像、
# 1 対多の写像、文脈依存の sigma) が高速路に落ちていないことは値でしか確かめられ
# ない。

# ASCII: 高速路。
print("The Quick Brown Fox 019_".upper())
print("The Quick Brown Fox 019_".lower())
print("The Quick Brown Fox 019_".casefold())
print("".upper(), "".lower(), "".casefold())

# latin-1 の上半分。U+00FF は U+0178 に上がるので幅が変わる = 高速路に入っては
# いけない。
print("ÿ".upper())
print(len("ÿ".upper()), "ÿ".upper() == "Ÿ")
print("é".upper(), "É".lower())

# 1 対多。ß は SS に、ǅ は文字数が変わらないが写像は別。
print("ß".upper(), len("ß".upper()))
print("ß".casefold())
print("ǅ".upper(), "ǅ".lower(), "ǅ".title())

# 文脈依存の sigma: 語末は ς、それ以外は σ。
print("Σ".lower())
print("ΑΣ".lower())
print("ΑΣΑ".lower())

# 混在: ASCII と非 ASCII が同じ文字列にあると一般経路。
print("abcÉ".lower(), "abcé".upper())
print("aKb".lower())
