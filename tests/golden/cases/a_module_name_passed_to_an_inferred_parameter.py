# An unannotated parameter is inferred from the call sites in the module, and
# the walk that reads them could resolve only literals and calls of functions
# it had already bound. A module-level NAME contributed nothing:
#
#     def show(s):
#         return s + "!"
#     x = "a"
#     print(show(x))
#     # function parameter 's' requires an annotation
#
# while `show("a")` in the same file compiled. Module assignments reach the
# shared symbol table as the EMITTER walks them, which is after the inference
# pre-pass, so the pre-pass now carries its own module scope.
#
# Why execution: an inferred parameter decides which BODY gets emitted, so the
# question is not whether the program compiles but whether it computes what
# CPython computes -- `average` returns a float only because its parameter is
# a list of ints, and `join` concatenates only because its is a list of str.
#
# ⛔ One type per inferred parameter: `scale(names, 2)` beside
# `scale(readings, 10)` is refused ("call arguments do not match the Callable
# contract"), because nothing here instantiates a second body. The names below
# are separated for that reason, not by accident.
#
# ⭐ The scope is built one statement at a time: a `for` header binds its
# target before the calls in its body are typed, and a name is invisible to
# the calls that precede its assignment.
#
# ⛔ `*args` and `**kwargs` are never inferred, before or after this: the
# pre-pass allocates a variable for each named parameter and none for the
# packed ones, so `def first(*args): return args[0]` is still refused. Binding
# one needs the call-site bridge to match a variadic tail against a variable,
# which is a different mechanism from the scope this repairs.
#
# ⛔ A `def` binds nothing here. The module scope SHADOWS the symbol table, so
# binding a function name would hand call sites a frozen signature instead of
# the one carrying this round's inference variables -- which is how a
# defaulted parameter widens (a_parameter_takes_the_type_of_its_default).


def scale(values, factor):
    out = []
    for v in values:
        out.append(v * factor)
    return out


def total(rows):
    n = 0
    for r in rows:
        n += r
    return n


def average(rows):
    if len(rows) == 0:
        return 0.0
    return total(rows) / len(rows)


def join(parts, sep):
    out = ""
    for p in parts:
        if out != "":
            out = out + sep
        out = out + p
    return out


readings = [1, 2, 3]
multiplier = 10
print(scale(readings, multiplier))
print(total(readings), average(readings))

for row in [[1], [2, 3]]:
    print(total(row), average(row))

names: "list[str]" = ["ann", "bob"]
print(join(names, "-"))
for name in names:
    print(join([name, name], "+"))
