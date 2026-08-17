# What this pins: `C.__name__`.
#
#     class C:
#         pass
#     print(C.__name__)
#     # attr.get type object has no static runtime attribute '__name__'
#
# The name is the one thing a type object cannot fail to know, and it is what
# `__repr__` bodies and error messages are written with. It folds to a string
# constant: the last dotted component of the contract name, so `int` answers
# "int" rather than "builtins.int", and a user class is its own answer.
#
# Why this needs to run rather than assert on a diagnostic: the value IS the
# fix, and a wrong fold (the qualified name, or a base's name where a subclass
# was asked) compiles and prints something plausible. The subclass pair below is
# the case that separates those.
#
# ⛔ This does NOT open the type-object surface. `print(int)`, `int is int`,
# `C.__class__` and `type(x)` are all still refused, and the fold is deliberately
# narrow: one dunder whose answer is static, not a runtime type object.
#
# ⛔ `list.__name__` is still "unresolved name 'list'": the container builtins are
# not bound as NAMES, which is a separate gap from this one. `int`, `str`, `float`
# and `bool` are bound, and they are the ones below.
#
# Every expected line is python3.14's.


class Registry:
    pass


class Base:
    pass


class Child(Base):
    def label(self) -> str:
        return Child.__name__ + ":" + Base.__name__


# --- a user class, at module scope and from inside a method ----------------
print(Registry.__name__)
print(Base.__name__, Child.__name__)
print(Child().label())


# --- the builtin scalars, where the contract name is qualified -------------
print(int.__name__, str.__name__, float.__name__, bool.__name__)


# --- it is an ordinary string afterwards ----------------------------------
name: str = Registry.__name__
print(name, name.upper(), len(name), name == "Registry")
print(name + "/" + Child.__name__)
print("-".join([Base.__name__, Child.__name__]))


# --- and it can be a dict key or a list element ---------------------------
labels = {Base.__name__: 1, Child.__name__: 2}
print(sorted(labels.items()))
names = [Registry.__name__, Base.__name__]
print(names, len(names), Registry.__name__ in names)
