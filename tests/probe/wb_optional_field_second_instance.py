# OPEN, and it PREDATES the Optional-field store repair of 2026-08-14 --
# measured identically on a binary built before it, so it is not that change's
# doing and not fixed by it.
#
#     released owned resource from builtin.unrealized_conversion_cast is used
#     after release (by call to '__ly_dealloc_Node')
#
# reported at the FIELD DECLARATION, not at any of the statements below it.
#
# BISECTED against ./build/bin/lyc:
#
#   two instances, NEITHER stored into ......... refused   <- this file
#   the same with no `name` field .............. refused
#   ONE instance, not stored into .............. OK
#   two instances, the FIRST stored into ....... OK
#   two instances, BOTH stored into ............ OK
#   two instances, both read, first stored ..... OK
#
# So it is not the store -- it is the second instance whose Optional field
# still holds the declaration's dead None placeholder when the object is torn
# down. One instance is fine, and storing anything into either one is fine.
#
# LOCATED, at runtime-lowering. The two instances' owned-local markers share
# SSA lanes:
#
#   %13:5 = ...cast %alloc,    %alloc_7,  %c1_i64_42, %11, %cast_43
#                                     {owned_local_object, ..._contract = "Node"}
#   %21:5 = ...cast %alloc_47, %alloc_56, %c1_i64_42, %11, %cast_43
#                                     {owned_local_object, ..._contract = "Node"}
#
# The first two lanes are each object's own storage and differ. The last three
# are the Optional field -- the constant tag naming the None member, and the
# dead placeholder `materializeNonOwningDeadObjectValue` built for it -- and
# they are THE SAME VALUES, because a constant and an inert placeholder are
# exactly what CSE merges.
#
# Physically that is fine: the None member owns nothing, so sharing costs
# nothing. It is the ownership walk that cannot live with it. Group identity is
# the ROOT (lane 0, distinct here), but "does this op use my group" is asked of
# ANY lane, so `__ly_dealloc_Node(%21...)` reads as a use of %13's resource --
# after %13 was released. Hence the diagnostic naming the marker rather than
# the placeholder, and hence a store fixing it: a store re-roots the field's
# lanes into that object's own expansion, and the sharing goes away.
#
# ⛔ So the repair is NOT to stop sharing the placeholder -- CSE would merge it
# again, and it is correct to share. Either the walk's "contains operand" test
# has to ignore lanes that own nothing, or a marker's inert lanes must not be
# part of what identifies it.
#
# It is why tests/golden/cases/optional_field_assignment.py stores into every
# instance it makes.
#
# differential: skip refused; the point is the refusal
from typing import Optional


class Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tag: Optional[str] = None


a = Node("a")
b = Node("b")
print(a.tag is None, b.tag is None)
