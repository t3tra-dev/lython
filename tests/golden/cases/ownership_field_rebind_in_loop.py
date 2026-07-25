# ループ本体でフィールドを再束縛し、ループ後に読む。CPython は `2` を出す。
#
# このファイルは errors/ から移ってきた。以前は `operand #N does not dominate
# this use` — MLIR verifier の生エラーで、Lython の診断ではない = 「最も早い
# 静的境界で拒否する」という原則に反する状態だった。フィールドの物理レーンを
# 差し替えると新しい SSA 名がループ本体に閉じ込められ、ループ後の使用を
# dominate しなかった。
#
# フィールド store が box16 スロットへの store になった今、ループ本体は新しい
# SSA 名を作らない (書き込み先が heap のスロットなので、支配関係の問題が
# 起こる余地がない)。同根だった片アーム if / 両アーム if / nested if / while /
# for / try 本体の 6 形すべてが同時に解消する。
class Holder:
    def __init__(self, xs: list[int]) -> None:
        self._xs: list[int] = xs


def mk() -> Holder:
    empty: list[int] = []
    return Holder(empty)


o = mk()
i: int = 0
while i < 3:
    fresh: list[int] = [i, i + 1]
    o._xs = fresh
    i = i + 1
print(len(o._xs))
