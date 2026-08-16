# What this pins: `if x:` where x is a UNION. Only Optional[T] was answered
# here -- None falsy, the single present member re-entering truthiness under
# the not-None guard -- and every other union fell through to the manifest
# evidence, which has nothing to answer with:
#
#     cfg = {"debug": True, "level": 3, "name": "app"}
#     if cfg["debug"]:
#     # static type !py.union<bool, int, str> does not provide manifest
#     # method '__bool__'
#
# A union has no class and no manifest contract of its own; it is decided by
# the TAG, which is how the same emitter already renders one for print. This is
# that dispatch for truthiness, and Optional[T] is now one case of it.
#
# Why this needs to run rather than assert on a diagnostic: the answer is
# per-member and every arm compiles. An empty string, a zero, an empty list and
# a None are the four falsy members here, and picking the wrong arm -- or
# defaulting the tag -- prints True where CPython prints False. Nothing but the
# printed value says which arm the tag chose.
#
# ⛔ A numeric member re-enters as `!= 0` rather than being refused the way a
# BARE numeric is ("implicit truthiness of int is rejected (Lython deviation
# from CPython)"). The deviation is about a numeric the writer could have
# compared explicitly, and no comparison covers `bool | int | str`. The last
# section is the control: a bare int is still refused.
#
# Every expected line is python3.14's.

# --- the record literal, which is where this shows up ---------------------
cfg = {"debug": True, "level": 0, "name": "", "tags": [1]}
print(bool(cfg["debug"]), bool(cfg["level"]), bool(cfg["name"]), bool(cfg["tags"]))
if cfg["debug"]:
    print("debug on")
if not cfg["level"]:
    print("level zero")
if cfg["tags"]:
    print("tagged")
if not cfg["name"]:
    print("unnamed")


# --- a heterogeneous list, all four falsy/truthy pairs ---------------------
xs = [0, "", 1, "a"]
print(bool(xs[0]), bool(xs[1]), bool(xs[2]), bool(xs[3]))


# --- a union a function returns -------------------------------------------
def pick(flag: bool) -> int | str:
    if flag:
        return 1
    return "a"


print(bool(pick(True)), bool(pick(False)))
v = pick(True)
if v:
    print("picked")


# --- Optional[T], which used to be the only shape answered -----------------
from typing import Optional


def text(x: Optional[str]) -> str:
    if x:
        return x
    return "-"


def number(x: Optional[int]) -> str:
    if x:
        return "y"
    return "n"


def items(x: Optional[list[int]]) -> str:
    if x:
        return "y"
    return "n"


print(text("a"), text(None), text(""))
print(number(3), number(0), number(None))
print(items([1]), items([]), items(None))


# --- a union WITH None and more than one present member --------------------
mixed: list[int | None] = [1, None, 0]
print(bool(mixed[0]), bool(mixed[1]), bool(mixed[2]))


# --- THE CONTROL: a bare numeric is still refused, so this compares --------
n = 3
print(n != 0, n == 0)
