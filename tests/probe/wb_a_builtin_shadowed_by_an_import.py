# Three builtins whose SPELLING is also a class are still not shadowed by an
# import, while five plain ones now are (golden cases/a_builtin_shadowed_by_an
# _import):
#
#     # helpers.py: def range(n): return [n, n + 1]
#     from helpers import range
#     for v in range(7): print(v)     # counts 0..6; CPython prints 7, 8
#
# MEASURED (2026-09-06, RelWithDebInfo, today's tree). The split is exactly
# "is the spelling bound as a CLASS in the type system":
#
#   len / abs / sum / max / min / repr .... CORRECT as of today
#   str("a") / int("a") ................... a LOUD type error naming the
#                                           builtin ("builtins.str has manifest
#                                           method '__init__' but no signature
#                                           that accepts ...") -- which is at
#                                           least not silent
#   range(7) .............................. STILL SILENT: the builtin runs
#   every one of the eight written in ONE file  correct
#
# ⭐ WHY THE CLASS SPELLINGS MISS. `programBindsName` now counts the canonical
# binding an import leaves, which is what disabled the builtin fast paths --
# but a call on `str` reaches the CLASS-INSTANTIATION path before that gate,
# and that path asks `types.lookupClass(name)`, where the builtin spelling is
# bound. The note there says "only a `def` introduces a competing callable
# under the same top-level name"; an import is the other way to write one.
#
# ⛔ AND ADDING IT THERE WAS MEASURED TO CHANGE NOTHING: `from helpers import
# str` never reaches that gate at all, so the repair is further up, wherever
# the builtin spelling is resolved to its class.
def unused() -> int:
    return 0


from a_module_of_shadowing_ranges import range

out: list[int] = []
for v in range(7):
    out.append(v)
print(out)
