# WHAT: `reversed(("a", 2, None))`: an element of a heterogeneous tuple
#   selected by a run-time index has no single representation.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   sequence __getitem__ cannot retain evidence element '!py.union<!py.contract<"builtins.int">, !py.contract<"builtins.str">, !py.literal<None>>' selected by a dynamic index
# FIXED 2026-10-30: a run-time index selection whose result is a union takes
#   a reference per member (retainUnionMemberValues), as a union read from a
#   payload does.
# CPython 3.14 expects: [None, 2, 'a']
print(list(reversed(("a", 2, None))))
