# probe: `erased_object_field_str.py` の対照 -- 同じ `object` 注釈フィールドに
#   **int** を入れる。
#   `c3de5e7` では **正しく動く** (`7` を出す)。`kernel/4a` では **loud 拒否**。
#   つまりこのセルは silent → loud ではなく **正しい → loud** で、意図的な受理範囲の
#   縮小である。int で動いていたのは値がインスタンスヘッダのワードに収まっていた
#   からで、**表現の偶然**であって設計ではない (収まらない str/float/list/dict は
#   すべて silent だった)。CLAUDE.md の「object / Any に対するランタイム操作は
#   実装しない」に照らせば、偶然動く 1 型だけを残すより全面拒否が原則に沿う。
#   **記録の要点**: この 2 件を対にして初めて「silent 4 型の修正」と
#   「正しい 3 形の喪失」が同じ変更の両面だと読める。片方だけでは、
#   純粋な改善にも純粋な退行にも見えてしまう。
# axes: width=object(erased, int payload) op=field flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   a type-erased `object` value cannot be stored in field 'class.v'; annotate the field with the concrete type it holds
# CPython 3.14 expects: 7


class Holder:
    def __init__(self, v: object) -> None:
        self.v: object = v


h = Holder(7)
print(h.v)
