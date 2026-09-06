# A `@classmethod` that a subclass overrides is refused when it is reached
# through a base-typed INSTANCE, in both spellings:
#
#     'tag' is overridden by a subclass of 'Base', so this call cannot be
#     resolved from the static type of the receiver
#
# MEASURED (2026-09-06, RelWithDebInfo, today's tree). The `@staticmethod` twin
# of this file dispatched as of today (cases/a_staticmethod_reached_through_a
# _base_typed_value), so what is left is the half that turns on `cls`:
#
#   Base.tag() / Sub.tag() (through the CLASS) ....... correct
#   an overridden @staticmethod through an instance .. correct (dispatched)
#   `m = x.s` for that staticmethod .................. correct (dispatched)
#   an overridden @classmethod through an instance ... this file
#   `m = x.tag` for it ............................... the same refusal
#
# ⭐ WHY `cls` IS THE WHOLE DIFFERENCE. The arms enumerate classes that
# REDECLARE the method, most-derived first, and each one calls through the
# CLASS it names. For a staticmethod that is exactly CPython's answer: a
# subclass that merely inherits the method runs its parent's body, which is the
# arm it lands in. For a classmethod the body is the same but `cls` is not --
# CPython binds the RUNTIME class, so `class C(B)` inheriting B's hook gets
# `cls is C` where the B arm would bind B.
#
# ⛔ SO IT IS NOT THE SAME CANDIDATE SET, which is what makes this a different
# repair rather than one more line in the same one. Covering it means an arm
# per SUBCLASS rather than per redeclaration -- every class derived from the
# receiver's, whether it redeclares anything or not -- and the arms are
# `isinstance` chains, so the cost is the whole subtree rather than the
# overriding slice of it. Refusing is the honest half in the meantime: a
# dispatcher binding the wrong `cls` is a silent wrong value, and the value
# spelling was exactly that until today.
class Base:
    @classmethod
    def tag(cls) -> str:
        return "base:" + cls.__name__


class Sub(Base):
    @classmethod
    def tag(cls) -> str:
        return "sub:" + cls.__name__


class Leaf(Sub):
    pass


xs: list[Base] = [Base(), Sub(), Leaf()]
print([x.tag() for x in xs])
