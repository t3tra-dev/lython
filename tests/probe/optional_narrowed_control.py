# probe: optional_unnarrowed_print.py の対照 -- `is not None` で絞り込む慣用形と
#   `d.get(k, default)` の非 Optional 形。**どちらも通る**ので、Optional の
#   表現自体は存在し、欠けているのは「絞り込まれていない Optional を object
#   位置に渡す」経路とその診断だけである、と切り分けられる。
# axes: width=optional op=narrowed-read flow=ifboth
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2 / missing / 0 / 2

d: dict[str, int] = {"a": 1, "b": 2}

v: int | None = d.get("b")
if v is not None:
    print(v)
else:
    print("missing")

w: int | None = d.get("zz")
if w is not None:
    print(w)
else:
    print("missing")

print(d.get("zz", 0))
print(d.get("b", 0))
