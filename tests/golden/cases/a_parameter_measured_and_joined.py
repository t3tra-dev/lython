# An unannotated parameter takes its type from the call sites, and a use in the
# body is a CHECK against that type -- except where the use declared a
# PROTOCOL, which was taken as the answer:
#
#     def show(args):
#         if len(args) == 0:
#             return ""
#         return ",".join(args)
#     print(show(["a", "b"]))
#     # !py.protocol<"Iterable", [builtins.str]> does not provide '__len__'
#
# `str.join` declares `Iterable[str]`, so the parameter was fixed AT the bound
# and `len` was then asked of a protocol. The same function without the `len`
# compiled, which is what says the protocol was standing in for an answer
# rather than checking one.
#
# Why execution: the parameter's type decides which body is emitted, and the
# router below has to route -- counting its arguments and joining them are the
# two things it does.
#
# ⭐ Only in that direction: a protocol-typed VALUE handed to an inferred
# parameter really is what the caller has, and binding the variable to it is
# the call site speaking.
#
# ⛔ `sorted`, `max`, `list()` and `enumerate` never had this: their contracts
# name a type PARAMETER, which binds through the name-keyed map rather than
# through unify. `join` is the one that names the protocol outright.

ROUTES = {}


def route(name, arity):
    ROUTES[name] = arity


def call(name, args):
    if name not in ROUTES:
        return "unknown:" + name
    if len(args) != ROUTES[name]:
        return "arity:" + name
    return name + "(" + ",".join(args) + ")"


def summarise(parts):
    n = len(parts)
    return ",".join(parts) + " (" + str(n) + ")"


route("add", 2)
route("ping", 0)
print(call("add", ["1", "2"]))
print(call("add", ["1"]))
print(call("ping", []))
print(call("nope", []))
print(sorted(ROUTES.items()))
print(summarise(["x", "y", "z"]))
print(summarise([]))
