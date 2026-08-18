# What this pins: `import os.path`, which in CPython binds `os` and reaches the
# submodule as an attribute of it.
#
#     import os.path
#     print(os.path.join("a", "b"))
#     # unsupported import 'os.path'
#     # (and, once the import was accepted: builtins.object does not provide
#     #  manifest method 'join')
#
# The driver decides which stdlib sources to compile from the import statements,
# and for a dotted name it requested only prefixes that are PACKAGE DIRECTORIES.
# `os` is a module (os.py, which does `import posixpath as path`), so nothing was
# requested for it, the emitter found no source module of that name, and the
# manifest fallback bound `os` to the erased object placeholder -- where `path`
# is not an attribute at all. `import os` on its own always worked, which is what
# made the shape look like an import-statement problem rather than a discovery
# one.
#
# Why this must run: what the repair changes is WHICH MODULE GETS COMPILED, and
# the only way to see that is to call something that lives in it. join, basename
# and splitext come from posixpath through os.path; dirname is here because the
# separator handling is where a half-loaded module would still answer.
#
# The other two spellings bind the SUBMODULE rather than the root, and they work
# by asking the root what it publishes under that name: `path` is a name inside
# os.py's own body (`import posixpath as path`), so the dotted form resolves to a
# namespace this emitter already knows how to bind. `from os.path import join`
# needed the same prefix request in the driver -- asking only for "os.path",
# which names no source, asked for nothing at all.
import os.path
import os.path as ospath
from os.path import join

p = os.path.join("a", "b", "c.txt")
print(p)
print(os.path.basename(p), os.path.dirname(p), os.path.splitext(p)[1])
print(os.path.join("/x", "y"), os.path.basename("/x/"))
print(ospath.join("a", "b"), join("c", "d"))
