# Returning an element of a container built in the SAME frame is refused:
#
#     def f() -> int:
#         xs = [1, 2]
#         return xs[0]
#     # released owned resource ... is used by function return
#
# The long-form analysis lives with the predicate that decides it
# (`retainEvidenceElement`, Runtime/Ops/GetItemOps.cpp): the literal MOVES its
# element's token into the container and leaves the owned-local marker behind,
# so the read borrows a reference the container now owns.
#
# RE-MEASURED (2026-09-07, RelWithDebInfo, today's tree). Two rows moved:
#
#   xs = [1, 2];   return xs[0] ............ released owned resource ...
#   t = (1, 2);    return t[0] ............. same
#   xs = [1.5];    return xs[0] ............ same
#   d = {"a": 1};  return d["a"] ........... returned with 1 additional
#                                            retained ownership token(s)
#   xs = [True];   return xs[0] ............ cannot adapt builtins.bool return
#                                            value to callable return ABI 0
#   xs = ["a"];    return xs[0] ............ correct (a str element)
#   xs = [[1]];    return xs[0][0] ......... correct
#   def f(xs): return xs[0] ................ correct (a parameter)
#   xs = [1, 2];   print(xs[0]) ............ correct
#   xs = [1, 2];   return xs[0] + xs[1] .... correct
#   def f(*args): return args[0] ........... CORRECT NOW (was "returned with 1
#                                            additional"): a borrowed group
#                                            whose producer is a retain-rooted
#                                            marker no longer takes a second
#                                            return retain
#   for x in xs: return x .................. CORRECT NOW (was "reaches function
#                                            exit without release")
#
# ⭐ AN INT ELEMENT IS NO LONGER EXEMPT. The note at the predicate said an int
# or a str element had no token to move; an int is box-fronted now and carries
# one, so the shape reaches a two-line function that any Python program might
# contain. `return xs[0] + xs[1]` in the same function is fine, which is what
# makes this look like a diagnostic problem rather than an accounting one.
#
# ⭐ THE BORROWED HALF OF THIS FAMILY IS FIXED (2026-09-07): when the stored
# value is one the frame only BORROWS -- a parameter, `*args`, anything derived
# from an entry argument -- the read's mint is the frame's only token and the
# return transfers it, so `insertBorrowedReturnRetains` must not add a second.
# What is left here is the OWNED half: the frame holds the element's birth
# token AND the read's mint, they alias, and no per-resource walk keyed on
# aliasing can tell "released the container's, returned the mint" from
# "released it and returned it".
#
# ⛔ SEVEN REPAIRS MEASURED, none right; all seven are recorded at the
# predicate. The last two are the useful ones: declining the borrow when the
# source was moved changes nothing (the token the return carries is not the one
# those predicates hand out), and minting the read's token AT the read moves the
# refusal onto the retained call result instead. The counts are right -- three
# references taken, one given back -- and what cannot follow them is the alias
# MODEL, which has one owner per entity where this shape has two.
def first() -> int:
    xs = [1, 2]
    return xs[0]


print(first())
