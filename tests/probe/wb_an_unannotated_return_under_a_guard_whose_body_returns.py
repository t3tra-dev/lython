# WHAT: An unannotated function returns from both arms of a None guard on a
#   LOCAL, the narrowed arm first:
#       if held is not None: return len(held)
#       else: return 0
#   The result type is read by a second walk (`inferredFunctionResult`) over
#   every `return`, against the symbol table the FIRST walk left at the end
#   of the body -- and that walk, seeing a body that ends in `return`, keeps
#   the guard's false side (`held: None`) for everything after the `if`. So
#   `len(held)` is asked of None and the function is refused. The mirror
#   image (`if held is None: return 0` / `else: return len(held)`) compiles,
#   by the same accident. A parameter is narrowed on the scope and compiles
#   either way; so does a generator, whose walk records each yield and return
#   where it stands.
# KIND: バグ (bug) -- loud, from the emitter
# FOUND: 2026-10-20, while writing the golden for a ternary yield on a local.
# CLASSIFICATION: loud 拒否 (診断)
#   static type str | None is a union, and '__len__' runs on whichever member
#   the value holds
# FIXED 2026-10-30. It had agreed since an earlier round only because the
#   first walk LEAKED the guard's narrowing (`held: str`) into the code after
#   the `if`; closing that leak brought the refusal back. Each `return` is now
#   typed where the walk stood (`returnTypeAt`), and the end-of-body table
#   answers only a return the walk could not type in place.
# CPython 3.14 expects: 2


def f(flag: bool):
    held: str | None = "bb" if flag else None
    if held is not None:
        return len(held)
    else:
        return 0


print(f(True) + f(False))
