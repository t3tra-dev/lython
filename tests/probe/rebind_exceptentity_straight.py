# probe: the except-bound exception ENTITY assigned to an outer binding; no loop
# axes: acquire=except width=wNexc op=rebind flow=straight
# CLASSIFICATION @ kernel/lane-dict 7cd3b94: 2 silent 誤実行 (prints the stale
#   pre-try value; also reproduces at main 1c3dfc4, so it predates one-laning)
# CPython 3.14 expects: boom
#
# The existing rebind_except_* probes read a FIELD out of an except-bound
# exception and are correct. This one assigns the exception itself outward, which
# is the case none of them reach.
kept: BaseException = ValueError("init")
try:
    raise ValueError("boom")
except ValueError as e:
    kept = e
print(str(kept))
