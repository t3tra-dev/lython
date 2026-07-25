# 呼び出しから受け取ったインスタンスのフィールド list を local alias 経由で
# 成長させる。CPython は `2` を出す。
#
# **このファイルは errors/ にあるべきではない。** RFC の段階 2 の受け入れ条件は
# これを cases/ (stdout `2`) にすることであり、ここに置いてあるのは移行が
# 途中であることの記録である。
#
# 移行前 (657f0d8): 同一バイナリ・同一入力で 3 通りの結果になった —
# rc=0 stdout `2` / rc=134 `Ly_IncRef observed non-positive refcount`
# (stdout が出ない = プログラムの出力自体が汚染される) / rc=133 stdout `2`。
# 原因は 2 つとも追跡単位が値リストだったこと: (1) 成長 primitive が
# フィールドのレーンを差し替えるとエンティティの同一性が失われ、解放が
# 「差し替え前のレーンの最終使用直後」= 関数の途中に置かれた、(2) その解放が
# deallocator に realloc 済みの古い payload ポインタを渡した。
#
# 移行後 (現在): root をキーにしてレーンを現在の展開へ進めるので (1) と (2) は
# 消え、代わりに unwind パスの残余が loud な診断になる — 解放が正しく後方へ
# 動いた結果、その手前の may-raise 呼び出しが「トークンを持ったまま unwind
# しうる」状態になり、その unwind cleanup は差し替え**前**のレーンで書かれ
# なければならない (差し替え後のレーンはまだ定義されていない)。エンティティ
# 1 個につき領域ごとの展開を持つ必要があり、それは物理 ABI を 1 レーンにする
# 段階 4 の作業である。
#
# したがってこの期待ファイルが検出すべきなのは:
#   - 黙って通るようになった (= mis-execution が戻った) → 退行
#   - rc=0 stdout `2` になった → 前進。このファイルを cases/ へ移すこと
class Node:
    def __init__(self, kids: list[int]) -> None:
        self._kids: list[int] = kids

    def add(self, v: int) -> None:
        ks: list[int] = self._kids
        ks.append(v)
        self._kids = ks


def leaf() -> Node:
    empty: list[int] = []
    return Node(empty)


n = leaf()
n.add(1)
n.add(2)
print(len(n._kids))
