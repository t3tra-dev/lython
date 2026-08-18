# What this pins: an imported module alias whose name a local also uses.
#
#     from os import path
#     print(path.basename("a/b.py"))
#     # <stdlib>/posixpath.py:221:12: unresolved runtime binding 'path.split'
#
# The failure was inside the compiler's own posixpath.py, at
# `comps = path.split("/")` -- a str method on `normpath`'s parameter, which is
# also called `path`. Binding the importer's alias put a canonical symbol named
# `path` in scope while that module was compiled, and the qualified-name route
# claimed the parameter's attribute chain. `import os` never collides, because
# nothing in the stdlib is named `os`; the collision is what the alias brings.
#
# So a local wins over an imported namespace, asked on the ROOT of the dotted
# name: `a.b.c` where `a` is a local is a local's attribute chain whatever `b` is,
# and a qualified symbol table cannot answer it.
#
# Why this needs to run rather than assert on a diagnostic: the shadowing decides
# WHICH function a call reaches. Resolving `path.upper()` on a str parameter to a
# module member would compile if the module happened to export `upper`, so the
# case below shadows the alias inside a function and calls both spellings in the
# same program.
#
# ⛔ `import os.path` and `from os.path import basename` are still "unsupported
# import": a dotted module name is a separate gap in the import resolver, with its
# own note in EmitterImports.cpp.
#
# Every expected line is python3.14's.

import math
from os import path


def shadowed(path: str) -> str:
    # `path` here is the parameter, not the module.
    return path.upper()


def also_shadowed(math: str) -> int:
    return len(math)


# --- the alias itself ------------------------------------------------------
print(path.basename("a/b.py"), path.dirname("a/b.py"))
print(path.join("a", "b"), path.splitext("x.py")[1])
print(path.normpath("a//b/../c"))

# --- the same name as a local, in the same program -------------------------
print(shadowed("x"), also_shadowed("abc"))
print(path.basename("c/d.txt"), math.floor(2.7))
