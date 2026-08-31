# `try: return [x] ... finally: ...` is refused: "return value type through
# try/finally is not implemented yet". Without the `finally` the same function
# compiles (fixed 2026-09-01, golden a_handler_returns_a_container), and with
# an `int` or a `str` return type the finally form compiles too.
#
# ⭐ WHAT THE FINALLY NEEDS THAT NOTHING ELSE DOES: the completion payload is a
# result of the `py.try` region, and the EXCEPTIONAL path has to yield one
# too -- `defaultCompletionValue` (Ops/TryOps.cpp) synthesizes it, and it can
# only do so for the types that have a static default: None, bool, int, float,
# str, object. A container has none, and the message says exactly that:
# "can only synthesize exceptional defaults for statically defaultable
# completion results".
#
# ⛔ AN EMPTY LITERAL IS NOT THE FIX AS WRITTEN. The synthesis runs inside the
# runtime lowering, so a `py.list` created there is a py op the pass has
# already gone past -- it would need the EMITTER to yield the inactive payload
# instead, which is where the shape is still known. The value's only job is to
# be released by the discard on the path that did not return, so an immortal
# dead placeholder would do as well as an empty list.
def first(table: "dict[str, int]", key: str) -> "list[int]":
    try:
        return [table[key]]
    except KeyError:
        return []
    finally:
        print("checked")


print(first({"a": 1}, "a"), first({"a": 1}, "z"))
