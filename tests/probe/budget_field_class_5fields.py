# probe: 5 フィールドのユーザークラスを別のクラスのフィールド型として使う。
#   当初は「段階 4a の box16 スロット化でコストが 1 + 5 = 6 になり予算 5 を
#   超えるので loud に退行する見込み」と書いていたが、**この予測は外れた**。
#   `kernel/4a` (`a36d881`) で実測: **通る** (素の実行 3/3 + libgmalloc)。
#   私の見積りは「int フィールド 1 個 = 1 スロット」を仮定していたが、実際は
#   contract 形の placeholder のぶんで **3** かかっていた。4a はその placeholder を
#   削除したので、**int フィールド N 個のクラスは 1 ハンドルに展開される**。
#   → 予算の境界は int フィールド数ではなく**参照型フィールドの本数**へ移った。
#     境界を押さえる golden は `kernel/4a` 側にある
#     (`errors/boxed_payload_too_wide_fields`、float 5 フィールド = 6 ハンドル)。
#     このブランチにはまだ無いので、統合後に参照が解決する。同じ統合で
#     `errors/boxed_payload_too_wide` (int 2 フィールド) は通るようになり
#     `cases/boxed_payload_int_fields_fit` へ移動している。
#   (この予測を測らずに書いたのは、本文書の運用ルール 2 の違反例そのもの。
#    見積りに使った「1 フィールド = 1 スロット」を一度も測っていなかった。)
# axes: width=wNcls(5 fields as a field type) op=construct+read flow=straight budget=1
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 1 5 / 6 10


class Five:
    def __init__(self, a: int, b: int, c: int, d: int, e: int) -> None:
        self.a: int = a
        self.b: int = b
        self.c: int = c
        self.d: int = d
        self.e: int = e


class Holder:
    def __init__(self, v: Five) -> None:
        self.v: Five = v


h = Holder(Five(1, 2, 3, 4, 5))
print(h.v.a, h.v.e)
fresh = Five(6, 7, 8, 9, 10)
h.v = fresh
print(h.v.a, h.v.e)
