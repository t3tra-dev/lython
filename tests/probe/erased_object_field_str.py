# probe: フィールドを `object` で注釈し、**str** を入れて読み戻す。
#   `c3de5e7` では **silent 誤実行** — 空文字列を出す (CPython は `ab`)。診断なし。
#   `kernel/4a` では **loud 拒否**: `a type-erased 'object' value cannot be stored
#   in field 'class.v'; annotate the field with the concrete type it holds`。
#   (初出時は "concrete class" だったが、int payload には class の指示が不適当で
#   5 形のうち 4 形で非 actionable になるため "concrete type" に修正された。)
#   payload が **インスタンスヘッダのワードに収まらない型** (str / float / list /
#   dict) はすべてこの silent でした (4 型で実測)。
#   k-4a の指摘で追加 (probe 集合にこの形が 1 件も無かった)。
#   対照は `erased_object_field_int.py` (int payload = c3de5e7 では正しく動く)。
# axes: width=object(erased, str payload) op=field flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 3 loud 拒否 (診断)
#   a type-erased `object` value cannot be stored in field 'class.v'; annotate the field with the concrete type it holds
# CPython 3.14 expects: ab


class Holder:
    def __init__(self, v: object) -> None:
        self.v: object = v


h = Holder("ab")
print(h.v)
