# A value whose static type is a union can now take an OPERATOR -- the tag is
# dispatched and every member answers or none does. The positions that are not
# operators still refuse.
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree), over `int | float`,
# `int | bool`, `int | str` and `list | str` producers:
#
#   u + 1, u * 2, u - 1, u / 2, u % 2, u ** 2, -u ... correct
#   1 + u, u += , u1 + u2 ............................ correct
#   u < 5, u == v, 0 < u < 5 ......................... correct
#   bool(u), str(u), repr(u), "{}".format(u) ......... correct
#   max(u, 1) ........................................ correct
#   len(u) ........................................... correct (2026-09-02)
#   u[0] ............................................. correct (2026-09-02)
#   [v for v in u] ................................... '__iter__'
#   hash(u) .......................................... runtime method receiver
#                                                      has no concrete contract
#   x in [u] ......................................... builtins.bool runtime
#                                                      object header (bool
#                                                      member only)
#   abs(u), round(u), int(u), float(u), sum([u, u]) .. each its own refusal
#
# ⭐ WHY THE LINE IS WHERE IT IS: `emitUnionMemberDispatch` turns the tag into
# a branch chain, and each CALLER has to be taught to use it -- the operator
# arms and `len` are, subscription, iteration and the numeric builtins are not.
# Each of those has its own emitter with its own ladder, and the type channel
# must not answer for a call whose emitter cannot build the tag test: the two
# would then disagree, and the value would be coerced to a member that may not
# be live.
#
# ⛔ `hash` and `in` are a different question again: both want ONE object
# handle for the whole value, which a union does not have (see
# wb_union_bool_member_in_a_container for the same wall on a container
# element).
#
# ⛔ AND ITERATION IS NOT A CALL. Dispatching `for v in u` per member would
# duplicate the LOOP BODY, and dispatching only the `__iter__` gives back an
# iterator whose type is itself a union -- which then needs a slot to live in
# across the loop, which is the storage wall this family keeps meeting. That is
# why the two members' answers joining is not enough here.
def mk(n: int):
    if n < 0:
        return "ab"
    return [1, 2]


print([v for v in mk(-1)], [v for v in mk(1)])
