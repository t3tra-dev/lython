# probe: a method stores into self's int | None field; the caller reads it back
# axes: acquire=self width=optional op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   runtime object header has invalid type 'i64'
# CPython 3.14 expects: 5
#
# RECLASSIFIED 2026-08-18: it prints 5 now, and LEAKS 52 B doing it -- the only
# leaking program in tests/probe (330 clean, 12 unmeasurable). The leak is the
# visible end of something larger, and the IR says it plainly:
#
#     %4:5 = call @mk()                     // the Box, owned
#     call @__ly_dealloc_Box(%4#0, ...)     // deallocated HERE, before set()
#     %5 = LyLong_FromI64(5)
#     Ly_IncRef(...) "builtins.int:class.f" // retained for a slot of a dead object
#
# The instance is released immediately after the call that produced it, because a
# union field is lane-spliced into the caller's own SSA expansion: `o.set()` and
# `print(o.f)` are answered from the folded value bundle, so the object HANDLE has
# no further use and the placement calls it dead. The store then retains the int
# for a slot nothing will ever release.
#
# Trigger, gridded 2026-08-18: the instance must come from a FUNCTION. Constructing
# `Box(None)` at module scope and storing is clean (0 B), with a union local or
# without; `mk()` plus any union-field store leaks, whether the stored value is a
# local (`fresh: int | None = 5`) or a literal (`self.f = 5`).
#
# The allocation site, from an LSan build (`lyc --fsanitize=leak`), is `__main__`
# -- the inlined `set()` body -- confirming it is the store's retain and not the
# constructor's.
#
# Not fixed: the release placement is the insertion pass's, and the store's slot is
# the object's SSA expansion, so making the object live past the store is the same
# lane-splice question the union field's other defects wait on.

class Box:
    def __init__(self, v: int | None) -> None:
        self.f: int | None = v

    def set(self) -> None:
        fresh: int | None = 5
        self.f = fresh


def mk() -> Box:
    v: int | None = None
    return Box(v)


o = mk()
o.set()
print(o.f)
