# A `@staticmethod` or `@classmethod` that a subclass overrides is refused when
# it is reached through a base-typed INSTANCE:
#
#     's' is overridden by a subclass of 'Base', so this call cannot be
#     resolved from the static type of the receiver
#
# MEASURED (2026-09-03, RelWithDebInfo, today's tree). The refusal is exactly
# the dispatcher's shape rule, and everything the dispatcher does cover is
# correct:
#
#   Base.s() / Sub.s() (through the CLASS) ............ correct
#   an overridden instance method through the base .... correct (dispatched)
#   an overridden @property through the base ......... correct (dispatched)
#   an overridden @staticmethod through an instance .. this file
#   an overridden @classmethod through an instance ... the same refusal
#
# ⭐ THE DISPATCHER RESTATES A SIGNATURE AND CALLS THROUGH IT, and its rule is
# that it may not GUESS one -- "a classmethod/staticmethod ... falls through to
# the refusal", says the note above `virtualDispatcherFor`. A staticmethod has
# no receiver parameter to restate, so the dispatcher would have to take one it
# then drops, and a classmethod's first parameter is the CLASS. Both are
# writable; neither is what the existing synthesis writes.
#
# ⛔ NOT a wrong answer, and the class-level spelling of both is correct today,
# which is the shape most programs use. What it costs is `x.s()` where `x` is
# base-typed -- rare enough that the refusal has stood, and recorded here so
# the next reader does not rediscover it.
class Base:
    @staticmethod
    def s() -> str:
        return "B"


class Sub(Base):
    @staticmethod
    def s() -> str:
        return "S"


print(Base.s(), Sub.s())
x: Base = Sub()
print(x.s())
