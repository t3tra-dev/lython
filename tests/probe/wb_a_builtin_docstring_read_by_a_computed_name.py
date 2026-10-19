# WHAT: `getattr(x, name).__doc__` where the name finds something whose
#   docstring is a builtin's -- object's own `__init__`, a str field -- raises
#   a TypeError that says so, where CPython answers the builtin's docstring.
#   No builtin keeps its docstring here; the arm is reached only if the run
#   time computes that name, so refusing it statically would refuse every
#   computed read (unittest's `getattr(self, methodName).__doc__` among them).
#   The literal spellings (`len.__doc__`, `int.__doc__`) are refused at
#   compile time: EmitterTest.ADocstringIsRefusedWhereItIsNotTheProgramsOwn.
# KIND: 未対応 (unsupported) -- loud, at run time
# FOUND: 2026-10-20, making docstrings readable for the unittest port.
# CLASSIFICATION: loud 拒否 (実行時の TypeError)
#   getattr(x, '__init__').__doc__ is not supported here: the docstring of
#   builtins.object's '__init__' is not retained
# CPython 3.14 expects: Initialize self.  See help(type(self)) for accurate signature.
class C:
    pass


name = "__init__"
print(getattr(C(), name).__doc__)
