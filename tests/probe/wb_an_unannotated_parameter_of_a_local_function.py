# OPEN. An unannotated parameter is inferred from call sites, and the pre-pass
# that does it covers TOP-LEVEL functions only (CLAUDE.md says so outright).
# A nested `def` is refused whatever it is called with, and a lambda -- which
# takes its parameters from an expected Callable rather than from the pre-pass
# -- is refused for some argument spellings and not others.
#
# MEASURED 2026-09-09, RelWithDebInfo, after the module-scope repair beside it
# (cases/a_module_name_passed_to_an_inferred_parameter). Every row is a program
# CPython runs; "refused" is `function parameter '<name>' requires an
# annotation`.
#
#   TOP-LEVEL def, argument spelled as
#     a literal ............................................. correct
#     a module NAME ......................................... CORRECT NOW
#     a module name with an annotation ...................... CORRECT NOW
#     a subscript of a module name .......................... CORRECT NOW
#     a module-level `for` target ........................... CORRECT NOW
#     a `for` target inside another function ................ correct
#
#   METHOD of a class, argument spelled as
#     anything ............................................. refused
#
#   NESTED def, argument spelled as
#     a literal ............................................. refused
#     the enclosing function's parameter .................... refused
#     a local ............................................... refused
#     a `for` target ........................................ refused
#
#   LAMBDA, argument spelled as
#     a literal ............................................. correct
#     the enclosing function's parameter .................... correct
#     a subscript, written at the call ...................... correct
#     a local assigned a literal ............................ correct
#     a local reached by `+=` ............................... correct
#     a local assigned a subscript .......................... refused
#     a local assigned the enclosing parameter .............. refused
#     a local assigned a call result ........................ refused
#     a `for` target ........................................ refused
#     inside a comprehension ................................ refused
#
# ⭐ THE LAMBDA'S RULE IS "A LOCAL WHOSE VALUE IS A LITERAL", not "a local":
# `show(names[0])` written at the call site resolves, and `x = names[0];
# show(x)` does not. Whatever computes the expected Callable reads the
# argument EXPRESSION and a literal-valued local, and nothing else -- which is
# the same shape the module scope had before the repair beside it, one scope
# further in.
#
# ⛔ A METHOD is the same boundary seen from the other side, and the one that
# costs most: `def __init__(self, text)` is how a class is written when nobody
# annotates it, and every field derived from such a parameter is `object`. Its
# call sites are `obj.method(...)`, whose callee is an Attribute rather than a
# Name, so reaching them is a different resolution from the one the pre-pass
# makes -- measured 2026-09-09 on a parser class whose `__init__` and `fail`
# were both unannotated.
#
# ⛔ The nested `def` is a stated boundary, not an oversight: inference runs in
# the module pre-pass, which walks `module.body` and finds top-level functions
# only. Extending it means deciding whether a nested def's call sites are its
# enclosing body (they are not, for one returned as a closure) -- a feature
# question, recorded rather than answered.
#
# ⛔ And two call sites of different types are refused for ANY of these
# ("call arguments do not match the Callable contract"): one inferred
# parameter is one type, because nothing instantiates a second body.
def outer(names: "list[str]") -> str:
    show = lambda s: s + "!"
    out = ""
    for item in names:
        out = out + show(item)
    return out


print(outer(["a", "b"]))
