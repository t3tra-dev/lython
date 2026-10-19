# WHAT: A str handed to a parameter declared `Iterable[str]` -- which it is --
#   and joined there: `"-".join(parts)` with `parts` the str "cd". The join
#   inside the function receives the str's own lanes where its manifest
#   overload for an iterable expects a length word: "cannot adapt
#   builtins.str to runtime input 2 of builtins.str.join". A list of str in
#   the same parameter works.
# KIND: バグ (bug) -- loud, from the lowering
# FOUND: 2026-10-20, writing the golden for a callable value whose dispatch
#   may reach an `Iterable[str]` parameter.
# CLASSIFICATION: loud 拒否 (診断)
#   cannot adapt builtins.str to runtime input 2 of builtins.str.join
# CPython 3.14 expects: c-d / a-b
from collections.abc import Iterable


def joiner(parts: Iterable[str]) -> str:
    return "-".join(parts)


print(joiner("cd"))
print(joiner(["a", "b"]))
