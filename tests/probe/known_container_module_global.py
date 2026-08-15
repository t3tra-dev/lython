# FIXED 2026-08-15. A container-typed module global read from a function was
# "unresolved name 'TABLE'", and so was `list`, `tuple`, `set`, `frozenset`
# and every stdlib contract -- only int/str/float/bool, bytes, a ctypes
# pointer and a user class had cells.
#
# ⭐ THE EXCLUSION WAS A STALE RATIONALE, NOT A MISSING MECHANISM. It read
# "their structural mutations reallocate the interior arrays through SSA
# rebinding, which a storage cell would go stale against", and that describes
# a representation this compiler stopped having. `builtins.mlir` says the
# current one, once per container: "a growth writes the new address THROUGH
# the handle, so every holder observes it with no further action and a
# mutation has nothing to rename. That is what lets ensure_capacity / extend /
# __setslice__ / __delslice__ be void and non-transferring." A cell holds the
# handle; the handle is what stays put. Measured before writing any code:
# allowing the five container contracts through made 50 appends from empty
# (six reallocations) read back correctly through a fresh cell load.
#
# ⛔ WHAT THE FIRST MEASUREMENT ALSO FOUND, and it had to be fixed first: the
# experiment turned `stdlib_pathlib` into a SILENT wrong answer -- four
# directory entries where CPython prints two. Not a container-global defect at
# all. `iterdir` has a local `names`, the program had a global `names`, and the
# importer's globals were in scope inside every imported module
# (`wb_global_shadows_stdlib_local.py`). Enabling container globals only made
# a live defect reachable by more programs.
#
# ⛔ WHAT IS STILL VALUE-BOUND, and why it is not the same list: an annotation
# that is not a CONTRACT (a union -- so isinstance narrowing keeps working on
# the module flow -- a protocol, a callable, `type[X]`, a tensor), and every
# UNannotated module name, which is the opt-in rule and predates all of this.
# `T = {"a": 1}` with no annotation is still unresolved from a function, where
# `N = 5` works because a name bound once to a literal re-emits the literal and
# a container literal cannot.
#
# golden: tests/golden/cases/container_module_globals.py (red-checked, and in
# the leak gate because the cell parks a retained reference).

TABLE: dict[str, int] = {"a": 1}


def look(k: str) -> int:
    return TABLE[k]


print(look("a"))
