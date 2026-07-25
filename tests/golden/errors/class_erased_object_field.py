# `object` と注釈したフィールドへの store は拒否される。
#
# フィールドのスロットは canonical な payload handle を持つ (word 1 = payload の
# class id、words 4+ = payload 自身の memref)。型消去された `object` は記述すべき
# 具体的な shape を持たないので、スロットに書けるのは「箱への箱」の間接であり、
# それを読み戻す側は payload と区別できない。
#
# 段階 4a より前、このフィールドはハンドル store 経路を通って**コンパイルが通り、
# 黙って古い値を読み続けていた** (`rebind_paramkept_w1obj` と同型の silent)。
# 拒否になったのはプロジェクトの原則どおりの前進である — `object` に対する
# ランタイム操作は実装しない、という方針とも一致する。
#
# 注意: **ユーザークラスで注釈したフィールドはこれに該当しない** (`self.f: Inner`
# は通る)。拒否されるのは注釈自体が `object` の場合だけである。
class Inner:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Holder:
    def __init__(self, v: object) -> None:
        self.f: object = v


h = Holder(Inner(1))
print("unreachable")
