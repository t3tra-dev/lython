# WHAT: `print(*xs)`: a starred sequence of unknown length reaches only the
#   *args of a function the program defines.
# KIND: 未対応 (unsupported) -- by design for now
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   a starred argument whose length is known only at run time can be passed here only to the *args of a function or method this program defines
# FIXED: 2026-10-29 -- print() collects its arguments as a tuple display and
#   writes each member as a piece (tryEmitPrintCall). Golden:
#   cases/print_writes_each_piece_before_rendering_the_next.
# CPython 3.14 expects: 1 2 3
xs = [1, 2, 3]
print(*xs)
