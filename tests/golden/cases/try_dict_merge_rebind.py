# `d |= other` inside try: the update desugar runs and the dict keeps both
# keys. This was a golden errors/ case for as long as the merge REBOUND the
# receiver -- the post-rebind token had no unwind release plan, and the
# rejection was loud rather than a silent revert of the mutation. One-laning
# builtins.dict deleted the rebind, so there is no post-rebind token to plan
# for and the case is ordinary.
#
# What still fixes the diagnostic it used to assert: errors/try_structural_rebind
# is the same program over `list`, which still rebinds, and still rejects.
d = {"a": 1}
try:
    d |= {"b": 2}
except ValueError:
    pass
print(d)
