# 呼び出しから受け取ったインスタンスのフィールド list を local alias 経由で
# 成長させる。CPython は `2` を出す。
#
# このファイルは errors/ から移ってきた。3 段階の履歴がある:
#   657f0d8: 同一バイナリ・同一入力で 3 通り — rc=0 `2` / rc=134
#     `Ly_IncRef observed non-positive refcount` (stdout が出ない =
#     プログラムの出力自体が汚染される) / rc=133 `2`
#   追跡単位を root にした後: unwind パスの残余で loud な拒否 (rc=1)
#   フィールド store を heap スロットにした後 (現在): rc=0 `2`
#
# 通るようになった理由は、フィールド store がインスタンスの SSA レーンを
# 差し替えるのをやめ、box16 スロットへの store になったこと。成長 primitive が
# payload を realloc してもエンティティの同一性は保たれ、解放は真の最終使用の
# 後ろに置かれる。そして「差し替え前のレーンで unwind cleanup を書く」という
# 要求 — 前段の loud な拒否の原因 — は、差し替えが存在しないので生じない。
#
# 検出すべき変化: 出力が `2` 以外になる (mis-execution の再来)、または再び
# 拒否に戻る。
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
