# 2 つの int フィールドを持つクラスを list の要素にする。CPython は `[Q]`。
#
# このファイルは errors/boxed_payload_too_wide から移ってきた。payload box が
# 運べるハンドルは 5 本で、以前このクラスは **7 本**に展開されていた —
# header 1 + int フィールドごとに 3 レーンのプレースホルダ。そのプレースホルダは
# 誰も読まない (int の値は最初からインスタンスヘッダのワードにある) のに幅を
# 消費し、予算超過でこのクラスはコンテナに入れられず、存在する `__repr__` も
# boxed dispatch から外れていた。
#
# int/bool フィールドがレーンを 1 本も取らなくなったので、このクラスは **1 本**に
# 展開される。予算が余るのは幅を縮めたからではなく、値でないものをレーンに
# 置くのをやめたからである (RFC §3.3 の議論と同じ形)。
#
# 予算そのものはまだ存在し、errors/boxed_payload_too_wide_fields がピン留めして
# いる。
class Q:
    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    def __repr__(self) -> str:
        return "Q"


print([Q(1, 2)])
