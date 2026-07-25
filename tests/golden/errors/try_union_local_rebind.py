# A union-typed local reassigned inside a try has no contract-typed slot to be
# promoted into, so neither the handler nor the continuation can observe the
# reassignment -- both would read the pre-try value. CPython prints 7 here.
# Rejected instead of answering None (this shape was a silent mis-execution
# before the storage promotion existed).
value: int | None = None
try:
    value = 7
except ValueError:
    pass
if value is None:
    print("none")
else:
    print(value)
