# 借用したコンテナの in-place 書き込み (`xs[0] = 42`) は通るが、成長
# (`xs.append`) は拒否される。成長 primitive が `transfer_args = [0]` +
# `owned_results = [0]` を持つ = 「エンティティを 1 個消費して別の 1 個を返す」
# 署名なので、借用にはその transfer を払う所有権がない。
#
# このファイルが固定するのは「拒否が loud であること」だけである。境界が
# 「レーンを再 root するか」に一致しているので、追跡単位がオブジェクト 1 個に
# なれば成長も借用のまま表せるようになる (その時点でこの期待は cases/ 側の
# 正常終了に置き換わる)。だが黙って通るようになってはならない。
def grow(xs: list[int]) -> None:
    xs.append(99)


ys: list[int] = [1, 2]
grow(ys)
print(len(ys))
