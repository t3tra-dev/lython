# `raise <a name>` -- an exception built into a local and raised, a parameter
# re-raised, a loop over prepared exceptions. Three DIFFERENT failures, all
# from the same place: `py.raise` TRANSFERS its operand to the raise
# primitive, and the frame's own reference to that value is not accounted for.
#
#   err = ValueError(m); raise err   (caught by the caller)
#     -> owned resource from @LyValueError_New result 0 is released or
#        transferred more than once on one CFG path
#     The frame still binds `err` when the handler resumes, so its
#     end-of-scope release runs on a reference the primitive consumed. The
#     verifier is RIGHT: this one would double-free.
#
#   def rethrow(e: Exception): raise e
#     -> borrowed entry argument 0 of @rethrow is released or transferred
#        without a prior retain
#     The mirror of the rule that already exists for a borrowed value RETURNED
#     (`valueGroupDerivedFromEntryArguments` at the func.return walk,
#     Passes/Ownership.cpp) -- a consuming call needs the same retain.
#
#   for exc in [ValueError("v"), KeyError("k")]: raise exc
#     -> runtime manifest has no .raise primitive
#     Note the EMPTY contract name: the union-typed element's bundle carries
#     no contract, so neither `manifest.primitive` nor the ancestor fallback
#     can name one. A different defect from the two above.
#
# ⛔ NEIGHBOURS THAT WORK, which is what makes the first two sharp: the same
# raise UNCAUGHT at module scope (nothing resumes to release it), `except X as
# e: raise e` (the handler's binding is not the frame's), and `raise problem`
# where `problem` is an optional local (the None arm keeps the token).
#
# ⭐ A BLANKET retain at the raise LOWERING is wrong and is why this is not a
# one-liner: `raise ValueError("x")` transfers a temporary, and a retain there
# leaks -- the raise never returns, so there is no point at which to release
# the extra reference. The retain belongs where ownership is decided, on the
# groups whose release is placed somewhere OTHER than this consume.
def rethrow(e: Exception) -> None:
    raise e


def build_and_raise(message: str) -> None:
    err = ValueError(message)
    raise err


try:
    build_and_raise("built")
except ValueError as e:
    print("caught built:", e)

try:
    rethrow(KeyError("passed"))
except KeyError as e:
    print("caught passed:", e)

for prepared in [ValueError("v"), KeyError("k")]:
    try:
        raise prepared
    except Exception as e:
        print(type(e).__name__, e)
