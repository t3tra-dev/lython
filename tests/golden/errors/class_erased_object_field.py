# `object` と注釈したフィールドへの store は拒否される。
#
# フィールドのスロットは canonical な payload handle を持つ (word 1 = payload の
# class id、words 4+ = payload 自身の memref)。型消去された `object` は記述すべき
# 具体的な shape を持たないので、スロットに書けるのは「箱への箱」の間接であり、
# それを読み戻す側は payload と区別できない。
#
# **この拒否は受理範囲の縮小でもある。両面を書く。** 段階 4a より前、このフィールドは
# ハンドル store 経路を通ってコンパイルが通り、payload の型で結果が割れていた
# (k-probe 実測、`erased_object_field_{str,int}` の対):
#
#   payload が str / float / list / dict → **silent** (空文字列 / 0.0 / [] / {})
#   payload が int                       → **正しく動いていた**
#
# つまり silent 4 型の修正と、正しく動いていた 1 型の喪失は同じ変更の両面である。
# int が動いていたのは値がインスタンスヘッダのワードに収まっていたからで、**表現の
# 偶然**にすぎない — 正しさが payload の幅に依存する経路をその 1 型のために残すより、
# 形ごと最早境界で拒否するほうが原則に沿う。CLAUDE.md が `object` / `Any` への
# ランタイム操作を実装しないと定めていることとも一致する。
#
# 注意: **ユーザークラスで注釈したフィールドはこれに該当しない** (`self.f: Inner`
# は通る)。拒否されるのは注釈自体が `object` の場合だけである。int payload 版は
# `class_erased_object_field_int` が押さえている。
class Inner:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Holder:
    def __init__(self, v: object) -> None:
        self.f: object = v


h = Holder(Inner(1))
print("unreachable")
