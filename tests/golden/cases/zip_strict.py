# What this pins: `zip(a, b, strict=True)` -- the spelling Python 3.10 added and
# the one a reader is told to prefer.
#
#     print(list(zip([1, 2], "ab", strict=False)))
#     # zip() takes no keyword arguments
#
# Every keyword was refused, which refused this one too. False is what zip
# already does; True adds the length check, and CPython says which argument
# differs and in which direction -- so the check is per argument, in argument
# order, and the first mismatch is the one reported.
#
# Why this must run: the answer is either the zipped pairs or a ValueError with
# a specific message, and both are runtime values. Longer and shorter are both
# here because they are different messages, and the equal-length case is here
# because it must NOT raise.
#
# ⛔ The literal is required: True and False are different EMITTED CODE, not a
# different value, so a computed flag is refused rather than guessed.
#
# ⛔ strict=True needs every argument to be indexable, including the first --
# which plain zip does not require of it. A leading iterator has no length to
# compare, and the refusal says so instead of comparing something else.
print(list(zip([1, 2], "ab", strict=False)))
print(list(zip([1, 2], "ab", strict=True)))

for a, b in zip([1, 2], [3, 4], strict=True):
    print(a, b)

try:
    print(list(zip([1, 2], "abc", strict=True)))
except ValueError as e:
    print("caught", e)

try:
    print(list(zip([1, 2, 3], "ab", strict=True)))
except ValueError as e:
    print("caught", e)
