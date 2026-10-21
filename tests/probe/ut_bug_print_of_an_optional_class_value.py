# WHAT: `print(u)` for `u: type[BaseException] | None`. The emitter lets it
#   through and the lowering refuses it ("unnarrowed ... cannot be used where
#   a concrete object is required"); `print(v)` for `v: type[BaseException]`
#   and `print(u)` after narrowing both work.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, making a class attribute hold a class value (unittest's
#   failureException).
# FIXED: 2026-10-21 -- a class member renders in a union's per-member chain
#   (canStringifyType / canConvertType had no arm for a class value).
# CLASSIFICATION @ lang-gaps e8c9b922 + the fix: 1 正しい
# CPython 3.14 expects: None / <class 'KeyError'>
u: type[BaseException] | None = None
print(u)
u = KeyError
print(u)
