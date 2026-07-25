# probe: 絞り込んでいない `T | None` を print する。CPython は値を出す
#   (`dict.get(k)` は typeshed どおり `int | None` を返す) が、lyc は
#   `runtime object header has invalid type 'i64'` で拒否する。
#   **診断が非 actionable** である点が要点 -- 「Optional は絞り込んでから使え」と
#   言っていない。同族の他の位置も同じ:
#     モジュールグローバルへの束縛 -> `module global assignment value has no
#                                     unbox.i64 primitive`
#     関数の戻り値              -> `cannot adapt return value to callable
#                                     return ABI 0 of f`
#   一方 `d.get(k, default)` (非 Optional を返す形) と、`if v is not None:` で
#   絞り込む慣用形は**どちらも通る** (optional_narrowed_control.py が対照)。
#   なお `d.get("b") + 1` の拒否は静的には健全なので loud で正しい。
#   既存の `wb_param_store_optional.py` / `wb_method_store_optional.py` が
#   Optional フィールドで loud なのと同じ族。
#   移植した contracts.py ハーネスで見つけた。
# axes: width=optional op=print-unnarrowed flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   runtime object header has invalid type 'i64'
# CPython 3.14 expects: 2

d: dict[str, int] = {"a": 1, "b": 2}
print(d.get("b"))
