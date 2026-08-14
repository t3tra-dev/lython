# OPEN, and now reported honestly. A field holding a TYPE object:
#
#     callable ABI type has no concrete runtime contract:
#     '!py.type<!py.contract<"Other">>'
#
# ⛔ It used to say something else, and the something else was FALSE:
# "'Box' inherits builtins.object.t, which Lython does not implement". There is
# no member `t` of object. The attribute-call path tried method dispatch, found
# no method `t`, and fell into the inherited-object refusal, whose predicate
# answers for ANY name once the class linearizes onto object -- so a field name
# arrives there looking like a missing dunder. Fixed 2026-08-15 alongside the
# callable-field family (tests/probe/wb_callable_field_store.py): a name the
# class declares as a field never reaches that refusal, and a field holding a
# type object is CALLED, since calling one constructs.
#
# ⭐ WHAT IS ACTUALLY MISSING is a runtime representation. A type object is
# materialised as an i64 class id at a call operand and nowhere else
# (`RuntimeBundle::Kind::TypeObject` in Runtime/Calls/Operands.cpp), and
# `runtimeShapeContractName` answers "" for `!py.type<...>`: the manifest
# declares `py.class @type` but no `ly.runtime.shape` for it, so there is no
# value shape a field could hold. The field store also expects
# `RuntimeBundle::Kind::Object` and would refuse a TypeObject bundle before
# reaching any layout question.
#
# So this needs three things and has none: a declared shape for the class id, a
# store/load path that accepts the TypeObject kind, and the call site reading
# the id back out of the field instead of from a constant.
#
# BISECTED (./build/bin/lyc):
#
#   Box(Other) then o.t(5).n .......... refused at the ABI   <- this file
#   Holder(Inner), never called ....... refused at the ABI (the STORE alone is
#                                       enough; s8_type_object_field)
#   the same field typed Callable ..... runs (callable_valued_field.py)
#
# The callable spelling running is what says the field machinery is fine and
# only the type object's representation is absent.
#
# differential: skip refused; the point is the refusal


class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, t: type[Other]) -> None:
        self.t: type[Other] = t


print(Box(Other).t(5).n)
