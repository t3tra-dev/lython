# Carryover (Wave 2): `d |= other` inside try (the update desugar) has no
# unwind release plan for the post-rebind token; loud, not a silent revert.
d = {"a": 1}
try:
    d |= {"b": 2}
except ValueError:
    pass
print(d)
