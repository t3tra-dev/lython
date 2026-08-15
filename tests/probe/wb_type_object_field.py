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
# ⭐ ATTEMPTED 2026-08-15 AND REVERTED, three layers in. Each edit moved the
# refusal one layer down, which is the useful part of the record:
#
#   1. `runtimeShapeContractName` answering "builtins.type" for `!py.type<...>`,
#      plus a `@LyType_Shape() -> i64` declaration in builtins.mlir
#        -> past "no concrete runtime contract", into
#           "runtime object header has invalid type 'i64'"
#   2. `primitiveFieldSlot` accepting builtins.type beside int and bool, since a
#      class id is exactly "a whole value in one i64" -- asked through
#      runtimeShapeContractName, because `!py.type<...>` is not a ContractType
#      and the plain name answers ""
#        -> past the header demand, into
#           "attribute value has no unbox.i64 primitive for field '_cls'"
#   3. `lowerAttrSet` materialising the class id as a constant instead of
#      unboxing, the same lookup a call operand does
#        -> NO CHANGE. The bundle reaching there is not
#           `RuntimeBundle::Kind::TypeObject`, and its `contractName()` prints
#           empty, so whatever the attr.set sees has already lost the kind.
#
# ⭐ AND THE FOURTH LAYER IS ANSWERED, cheaply, which changes the shape of the
# whole item. The attr.set's value is not a `py.type.object` result at all:
#
#     py.attr.set %arg0["_cls"] = %arg1 ... : !py.contract<"Holder">,
#                                             !py.type<!py.contract<"Inner">>
#
# `%arg1` is `Holder.__init__`'s PARAMETER. `lowerTypeObject` makes a
# `RuntimeBundle::typeObject` for a `py.type.object` op, and nothing makes one
# for an entry argument, so the store was never going to see the kind.
#
# So this is not "a field cannot hold a type object". It is that `type[X]` has a
# runtime representation at a CALL OPERAND and nowhere else -- not in a
# parameter, not in a field, not in a local. Four layers found and the parameter
# ABI is only the next; expect a read side and a call site behind it. Scope it as
# "give type[X] a representation end to end", not as a field repair. Reverted rather than left in: three layers of gate changes that
# still refuse the program are untested surface, which is the same rule that
# sent back the `consumeSites` variant and the insertion-block walk today.
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
