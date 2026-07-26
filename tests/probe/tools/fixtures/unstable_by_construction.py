# Not a probe. A fixture that is UNSTABLE BY CONSTRUCTION, so a tool whose
# healthy answer is "nothing was unstable" can be shown to be able to say
# otherwise.
#
# It prints the default object repr, which embeds the instance's address. That
# differs between runs of the same binary, and differs from CPython's, and both
# facts are correct behaviour -- there is nothing here to fix, which is the
# point. A tool's domain test must not depend on a defect that a later stage
# repairs: `alias_read_mutate_nowriteback_dict` was the corpus's only genuinely
# nondeterministic input, and stage 4b's interior-view repair removed it.
#
# What this does NOT cover: a real use-after-free's distribution of faces
# (ok / silent / abort / signal). Instability of the OUTPUT is enough to exercise
# the comparison, not enough to rehearse the shapes it was built to classify.
# Those still need a broken tree; see the rebuild recipe in leak.py.


class Thing:
    pass


print(Thing())
