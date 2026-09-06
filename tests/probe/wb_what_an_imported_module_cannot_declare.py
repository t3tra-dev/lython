# Which DECLARATION kinds survive the trip from an imported module into the
# program that imports them. Every failure recorded here is loud; the probe
# exists because "an imported module is a second-class module" is a claim that
# needs a list behind it, and because half the things that look like module
# boundary defects are not.
#
# MEASURED (2026-09-03, RelWithDebInfo, today's tree). A 20-way sweep declared
# one kind per module and used it from a second file; every case was then
# rewritten as ONE file, which is what says whether the boundary is the cause:
#
#   CORRECT ACROSS THE BOUNDARY
#     def / default argument / generator def
#     @property / @staticmethod / @classmethod / @dataclass
#     Exception subclass, class attribute read through an instance
#     int / str / bool module constant
#
#   ONE FILE IS RIGHT, ACROSS THE BOUNDARY IS NOT -- the boundary is the cause
#     Enum subclass ......... CORRECT as of 2026-09-06 (`desugarEnumClasses`
#                             now runs on every imported module before the main
#                             one, and an imported class attribute is real
#                             storage, which is what a member needs to BE an
#                             instance; golden cases/an_enum_declared_in_an
#                             _imported_module)
#     module-level lambda ... module 'm' has no attribute 'DOUBLE' in this
#                             runtime (annotated `Callable`, so the one-file
#                             spelling of it compiles and answers 2) -- the
#                             ONLY one left on this list
#     float / None constant ..CORRECT as of 2026-09-04 (the literal channel
#                             gained a float arm and a None arm)
#     container constant .... CORRECT as of 2026-09-06 (a container gets the
#                             module global CELL the main module's own gets,
#                             filled at the start of `__main__` in import
#                             order; golden cases/an_imported_container
#                             _constant_is_one_storage). Only when the element
#                             type is resolved: an erased `[]` has no storage
#                             the readers can agree on and keeps the refusal.
#
#   BOTH FAIL -- the boundary is NOT the cause, and neither is a finding here
#     nested class (`m.Outer.Inner()`) .... !py.type<...> does not provide a
#                                           manifest __call__ (also in one file)
#     generic def over `list[T]` .......... refused in one file too
#
# ⭐ THE TWO REAL ONES HAVE NO SYMBOL TO BIND. `bindSourceModuleNamespace`
# walks the module's top level and binds what it finds BY NAME: a def gets a
# canonical callable, a class gets a contract. A lambda is an expression bound
# by an assignment that walk reads only for `alias = name`, and an Enum's base
# is resolved where the class is EMITTED rather than where it is bound. Two
# different missing bindings, not one mechanism.
#
# ⛔ The three that read as boundary defects and are not were measured, not
# reasoned about: `Outer.Inner()` and the `list[T]` generic refuse in one file
# with the same shape of message. Writing them down here is the point -- the
# next module-boundary fix should be checked against this list rather than
# against the memory of a sweep.
import a_module_with_declarations as decls

print(decls.Color.RED.value)
print(decls.DOUBLE(1))
