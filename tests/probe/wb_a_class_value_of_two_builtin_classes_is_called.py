# WHAT: a name holding one of two builtin classes, chosen at run time, is
#   called -- `conv = int if c else float; conv(text)` -- and so is each
#   class of a list of them, the try-each-converter idiom.
# KIND: 未対応 (unsupported) -- loud
# FOUND: 2026-10-28, sweeping the union axis; measured again 2026-10-29.
# CLASSIFICATION @ lang-gaps (2026-10-29): 3 loud 拒否 (診断)
#   calling through a class value that may name any class is not supported:
#   each class it could name would be an arm
# The classes share no source base, so their join is `type[object]`
# (TypeSystem::join), a class value that may name any class, and a call
# through it has no arms to dispatch to. Classes WITH a source base join at
# that base's class value and are called through the subclasses it has, which
# works (`Dog if c else Cat`).
# What a union of the class values instead would need, measured 2026-10-29:
#   - a slot read that tells the classes apart. Every class value's box names
#     `type`, so `xs: list[type[int] | type[str]] = [int, str]` read `int` back
#     as `str`, silently, and so did `t[i]` over `(KeyError, IndexError)`. The
#     read now asks the held type object (LyType_IsSubtype); FIXED.
#   - `k is int` over a union, which is refused (wb_identity_against_a_union_
#     member) where `type[object]` answers it at run time: a list of classes
#     joined as a union would lose that spelling. That is what is left.
# FIXED 2026-10-30 for the call: classes with no common base join as a union
#   of their class values (`k is int` over one is answered now), so `conv =
#   int if c else str; conv("5")` and `[k("3") for k in [int, float]]` run.
#   What still refuses this file is `return kind(text)` inside a `try`:
#   `first_that_parses` returns `int | float | str`, and a union is not yet
#   carried out of a `try` ("return inside try is not implemented yet").
# CPython 3.14 expects: "5 str", "3 3.0 3.5", "1.5"
import sys

conv = int if len(sys.argv) > 5 else str
print(conv("5"), conv.__name__)


def first_that_parses(text: str) -> int | float | str:
    for kind in [int, float, str]:
        try:
            return kind(text)
        except ValueError:
            pass
    return text


print(first_that_parses("3"), first_that_parses("3.0"),
      first_that_parses("3.5"))
print(first_that_parses("1.5"))
