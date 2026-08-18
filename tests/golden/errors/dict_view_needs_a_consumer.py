# What this pins: a dict view bound to a name.
#
#     d: dict[str, int] = {"a": 1}
#     ks = d.keys()
#     # runtime manifest has no builtins.dict.keys method
#
# A sentence about the manifest for a program that did nothing to it, while every
# CONSUMING spelling works -- `len(d.keys())`, `sorted(d.keys())`,
# `list(d.keys())`, `for k in d.keys()` -- because each of those unwraps the view
# before emitting it. What has no representation is the view as a VALUE.
#
# Refused rather than snapshotted. CPython's view tracks later mutations of the
# dict and `list(d.keys())` does not, so binding a list where the program asked
# for a view is a silent wrong answer the moment anything inserts. The message
# names the consuming positions and the snapshot spelling, and says what the
# snapshot gives up.
d: dict[str, int] = {"a": 1}


ks = d.keys()
print(len(ks))
