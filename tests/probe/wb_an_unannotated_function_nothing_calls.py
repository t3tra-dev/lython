# OPEN, and a feature-boundary question rather than a mechanism one. An
# unannotated parameter takes its type from the call sites in the module, so a
# helper NOTHING calls has no evidence and the whole module is refused:
#
#     def keys_of(table):
#         out = []
#         for k in table:
#             out.append(k)
#         return out
#     # function parameter 'table' requires an annotation
#
# CPython runs that program: it compiles the body and never executes it. This
# is the last refusal left in a corpus of ~60 realistic programs written for
# this compiler, and the shape is ordinary -- a helper written before the code
# that will use it.
#
# MEASURED 2026-09-09, RelWithDebInfo:
#
#   never mentioned .................................... refused
#   mentioned as a VALUE (`f = helper`), never called ... refused
#   called only inside `if False:` ...................... correct
#   an unannotated METHOD nobody calls .................. refused
#   a recursive function nobody calls from outside ...... refused
#   the same function with annotations .................. correct
#
# ⭐ A DEAD BRANCH IS STILL A CALL SITE, which is the line the pre-pass draws
# today: it reads call sites syntactically, so `if False: helper([1])` types
# the parameter and a program with no mention at all does not. Whatever decides
# this has to say why those two differ, because to the READER they do not.
#
# ⛔ The repair that suggests itself -- skip emitting a top-level unannotated
# function nothing references, and say nothing about it -- is a decision to
# DROP code the reader wrote, not a fix to a mechanism. It is safe inside this
# compiler's model (no `globals()`, and an imported module's functions are
# refused for a different reason), and its failure direction is an "unresolved
# name" at a reference the scan missed rather than a wrong answer. Recorded
# rather than taken: what a compiler does with code it cannot type and nobody
# runs is the user's call.
#
# ⛔ The diagnostic is right, at least: "function parameter 'table' requires an
# annotation" comes first, naming the parameter and the line. The
# `!py.infervar<0> does not provide manifest method '__len__'` that follows it
# is the body walk reading the same unresolved parameter, and names the
# compiler rather than the program.
def keys_of(table):
    out = []
    for k in table:
        out.append(k)
    return out


print("ok")
