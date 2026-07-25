# ループ本体でフィールドを再束縛すると MLIR verifier の生エラーで落ちる。
# フィールドの物理レーンを差し替える (再 root) と新しい SSA 名がループ本体に
# 閉じ込められ、ループ後の使用を dominate しない。
#
# 期待している exit code と stderr は「Lython の診断ではない」ことを記録して
# いる: `operand #N does not dominate this use` は MLIR verifier の出力であり、
# 「最も早い静的境界で診断を出して拒否する」という原則に反する。したがって
# この期待ファイルは 2 通りの意味で変化を検出する — 黙って通るようになった
# ときと、Lython 自身の診断に置き換わったときの両方である (後者は前進なので
# 期待を書き換えてよい)。
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
