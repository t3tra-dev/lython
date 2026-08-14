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
# WHERE TO LOOK: `materializeNonOwningDeadObjectValue` builds the placeholder
# for the inactive members, and the tag is set to name a member that owns
# nothing (Runtime/ABI/RuntimeABI.cpp, "A dead union has to name a member that
# owns nothing"). Two instances built the same way get the same placeholder
# values, so the question is whether the two objects' teardowns are reaching
# ONE resource -- the diagnostic naming `__ly_dealloc_Node` on a
# `builtin.unrealized_conversion_cast` (the owned-local marker) is consistent
# with that.
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
