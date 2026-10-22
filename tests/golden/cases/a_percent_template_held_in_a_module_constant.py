# WHAT: %-formatting whose template is a module constant -- unittest's
# `DIFF_OMITTED % len(diff)` -- with an escape and a quote in it, read in a
# function and at module level.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: only a literal written in
# place was a template ("%-formatting requires a compile-time format string
# here"); the constant's text is the template now, and whether its escapes
# survived is seen only in the formatted string.
DIFF_OMITTED = ('\nDiff is %s characters long. '
                'Set self.maxDiff to None to see it.')
QUOTED = 'say "%s"\t%d'


def omitted(n: int) -> str:
    return DIFF_OMITTED % n


print(repr(omitted(236)))
print(QUOTED % ("hi", 3))
