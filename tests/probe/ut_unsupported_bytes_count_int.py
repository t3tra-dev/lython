# WHAT: `bytes.count(int)` (counting one byte value): the contract has only
#   the bytes-argument signature.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   static type !py.contract<"builtins.bytes"> has manifest method 'count' but no signature that accepts (!py.contract<"builtins.int">)
# CPython 3.14 expects: 2
print(b"abca".count(97))
