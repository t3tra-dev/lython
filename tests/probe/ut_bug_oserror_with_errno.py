# WHAT: `OSError(2, "...")` is refused with "attr.get object type has no
#   class schema" -- a message about the compiler, not the program. CPython
#   builds a FileNotFoundError with errno 2 and str "[Errno 2] ...".
# KIND: バグ (bug) -- loud, with an internal message
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   attr.get object type has no class schema
# CPython 3.14 expects: FileNotFoundError 2 [Errno 2] No such file or directory
e = OSError(2, "No such file or directory")
print(type(e).__name__, e.errno, str(e))
