# probe: alias_module_level_dict.py の list 版 (モジュール直下で読み出し・成長・
#   書き戻し)。dict 版が silent になるのに対し、list 版は非決定的に silent /
#   正常終了 / abort に分かれる (libgmalloc では確実に死ぬ)。
# axes: acquire=call width=w3list op=alias-read+grow+writeback flow=straight scope=module
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 1
# 観測: 8 回実行: 正常終了 5 (正解を出す) / abort(refcount) 3。libgmalloc: SIGSEGV (決定的)


class Node:
    def __init__(self, v: list[int]) -> None:
        self._f: list[int] = v


def leaf() -> Node:
    v: list[int] = []
    return Node(v)


n = leaf()
ks: list[int] = n._f
ks.append(1)
n._f = ks
print(len(n._f))
