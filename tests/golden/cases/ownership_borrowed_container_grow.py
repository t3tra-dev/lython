# 借用したコンテナの成長 (`xs.append`) が借用のまま表せることを固定する。
# 期待値は CPython のもの。
#
# **これは `errors/` から移ってきたケースである。** 以前は拒否されていた:
# 成長 primitive が `transfer_args = [0]` + `owned_results = [0]` を持つ =
# 「エンティティを 1 個消費して別の 1 個を返す」署名だったので、借用には
# その transfer を払う所有権がなかった。
#
# 移動の理由は「検出されなくなった」ではなく「存在しなくなった」である:
# `builtins.list` が 1 レーン (`memref<9xi64>`) になり、
# `LyList_EnsureCapacity` は新しい items base をハンドル越しに書くだけの
# void になったので、`builtins.list` を宣言する境界の `transfer_args` は
# 0 件になった。払うべき transfer が無ければ、未払いを拒否する規則は
# 発火しようがない。移動前のファイル自身が「追跡単位がオブジェクト 1 個に
# なれば成長も借用のまま表せるようになる (その時点でこの期待は cases/ 側の
# 正常終了に置き換わる)」と予告していたのは、まさにこの移動である。
#
# 判別子: run_case.py は `--release` を渡さない (verifier は有効)。
# 対照は同一実行内にある — `tests/probe/known_borrowed_set_add` は
# 同じ診断で LOUD のまま (set はまだ transfer する)。
def grow(xs: list[int]) -> None:
    xs.append(99)


ys: list[int] = [1, 2]
grow(ys)
print(len(ys))
