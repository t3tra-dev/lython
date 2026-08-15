# FIXED 2026-08-15. A module global of THIS program was in scope inside every
# imported module, where a LOCAL of the same spelling then answered at the
# global's type. Run this file's shape on the old binary and the diagnostic is
#
#     <stdlib>/pathlib.py:264:8: emit error: static type
#     !py.contract<"builtins.int"> does not provide manifest method 'sort'
#
# against a file the program never wrote, naming neither the global nor the
# collision. `iterdir` has a local `names`; declaring ANY annotated module
# global called `names` -- of any type -- broke it.
#
# ⭐ THE COMMENT AT THE SITE ALREADY SAID WHAT THE RULE IS, and the code did
# two thirds of it. `emitClassMethodInline` clears `values` and calls
# `types.isolateScopes()` under the note "A method of a source-module class
# executes under ITS module's globals (Python scoping), not the use site's".
# The module-scope value bindings live in THREE MORE maps -- `moduleGlobals`,
# `moduleConstantBindings`, `primitiveConstants` -- and those were not swapped,
# so `isModuleGlobalRead` kept answering for the importer. Named once as
# `ModuleEmitter::ImporterModuleScope` and applied at all three places that
# emit another module's code.
#
# ⛔ AND THE THIRD PLACE IS WHY THE FIRST TWO REPAIRS DID NOTHING. The obvious
# reading is that an imported module's code is emitted by
# `emitSourceModuleDeclarations`, so that is where the isolation was put first
# -- no change. Then `collectModuleGlobals` was moved to run after it -- no
# change either. A stdlib METHOD is emitted at neither: the emitter INLINES it
# at the call site, inside `__main__`, with only `sourceName` swapped. That is
# also why the local was never bound at all in the crashing spelling: at module
# scope `isModuleGlobalWrite` is true, so `names = os.listdir(...)` wrote the
# program's cell instead of binding a local, and the next line read the cell.
#
# ⛔ MEASURING THIS BY GREPPING THE STDLIB FOR LOCAL NAMES UNDERSTATES IT. The
# collision needs no rare name: `key`, `result`, `n`, `index`, `value`, `parts`
# and `i` are all locals of `collections`/`string` methods, and `key` alone
# appears in 19 of them.
#
# ⛔ WHAT IS STILL TRUE, and is a different question: an imported module has no
# global CELLS of its own. Its module-level annotated assignments are bound as
# literal constants (`bindSourceModuleLocals`), so a stdlib module cannot hold
# mutable module state. Nothing here changes that; the fix only stops the
# importer's cells from standing in for it.
#
# golden: tests/golden/cases/global_does_not_reach_imported_module.py
# (red-checked), which collides 21 names across Counter, OrderedDict and
# Template.

from collections import Counter

# `key` and `result` are locals of Counter.__eq__ and Counter.__add__.
key: str = "not-a-counter-key"
result: int = 7

counts: Counter[str] = Counter()
counts.update(["a", "b", "a"])
print(counts.most_common(1))
print(key, result)
