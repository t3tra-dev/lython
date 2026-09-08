# OPEN, and it is the last half of the classmethod dispatch. Calling one
# through a base-typed receiver now dispatches
# (cases/a_classmethod_reached_through_a_base_typed_value); READING it first
# does not:
#
#     m = x.tag       # x: Base, and Sub overrides tag
#     print(m())      # 'tag' is overridden by a subclass of 'Base', so this
#                     # call cannot be resolved from the static type
#
# MEASURED 2026-09-09:
#
#   x.tag() on a base-typed receiver ............... correct (dispatched)
#   x.make(2) with arguments ....................... correct
#   x.build() constructing through `cls` ........... correct
#   an INHERITED classmethod (`class Sub(Base): pass`) correct -- it was a
#                                                    silent `Base` before
#   Base.tag() / Leaf.tag() through the class ...... correct
#   m = x.tag; m() ................................. this file
#   the same value spelling for a @staticmethod .... correct (dispatched)
#   a classmethod GENERATOR that reads `cls` ....... refused, and it was a
#                                                    silent ['Base'] before
#   a `*args` classmethod that never reads `cls` ... correct (no dispatcher is
#                                                    asked for, so the vararg
#                                                    the arms cannot restate
#                                                    costs nothing)
#
# ⭐ WHY THE STATICMETHOD'S VALUE SPELLING WORKS AND THIS ONE DOES NOT. The
# value spelling routes through the same dispatcher, and for a staticmethod the
# dispatcher IS the callable: no receiver, no `cls`, so a reference to it
# answers. A classmethod's arms each call `Candidate.tag()`, so the dispatcher
# takes the receiver as its first parameter -- and a bare reference to it has
# nothing to bind that parameter to. What is missing is a bound object over the
# dispatcher, which is the same shape
# wb_a_bound_method_off_a_builtin_instance needs for a manifest method.
#
# ⛔ NOT the arms. They are right: `Leaf.tag()` runs Sub's body with
# `cls is Leaf`, which is what the golden pins.
class Base:
    @classmethod
    def tag(cls) -> str:
        return "base:" + cls.__name__


class Sub(Base):
    @classmethod
    def tag(cls) -> str:
        return "sub:" + cls.__name__


xs: "list[Base]" = [Base(), Sub()]
for x in xs:
    m = x.tag
    print(m())
