# WHAT: `print(u)` for `u: type[BaseException] | None`. The emitter lets it
#   through and the lowering refuses it ("unnarrowed ... cannot be used where
#   a concrete object is required"); `print(v)` for `v: type[BaseException]`
#   and `print(u)` after narrowing both work.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, making a class attribute hold a class value (unittest's
#   failureException).
# CLASSIFICATION @ lang-gaps 51e1e5ed + the class-attribute fix: 3 loud 拒否 (診断)
#   unnarrowed !py.union<!py.literal<None>, !py.type<!py.contract<"builtins.BaseException">>> cannot be used where a concrete object is required
# CPython 3.14 expects: None / <class 'KeyError'>
u: type[BaseException] | None = None
print(u)
u = KeyError
print(u)
