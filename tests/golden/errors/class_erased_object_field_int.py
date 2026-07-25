# `object` 注釈フィールドの拒否のうち、**段階 4a より前は正しく動いていた**側。
#
# CPython は `7` を出し、`c3de5e7` の lyc も `7` を出していた (k-probe 実測)。
# 同じ形で payload が str / float / list / dict のときは silent 誤実行だったので、
# int だけが通っていたのは値がインスタンスヘッダのワードに収まるという**表現の
# 偶然**である。正しさが payload の幅に依存する経路をこの 1 型のために残すより、
# 形ごと拒否する方を選んだ (`class_erased_object_field` の docstring に理由)。
#
# このファイルが存在する理由は、縮小をテストスイートに記録しておくことである。
# 拒否でなくなったとき (= 型消去されたフィールドを本当に実装したとき) に気づける。
class Holder:
    def __init__(self, v: object) -> None:
        self.f: object = v


h = Holder(7)
print(h.f)
