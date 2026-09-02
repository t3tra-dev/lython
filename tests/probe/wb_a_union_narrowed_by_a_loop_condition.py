# A union narrowed by the CONDITION OF A LOOP is not narrowed on the way out:
#
#     cannot adapt !py.union<!py.contract<"builtins.int">,
#     !py.contract<"builtins.str">> return value to callable return ABI 0 of f
#
# MEASURED (2026-09-03, RelWithDebInfo, today's tree). The `while` is the only
# ingredient; every other spelling of the same narrowing is correct:
#
#   if isinstance(v, int): v = "done"   then return v ....... correct
#   while isinstance(v, int): v = "done", `-> int | str` .... correct
#   while ..., then `if isinstance(v, str): return v` ....... correct
#   while isinstance(v, int): v = "done", then `-> str` ..... this file
#
# ⭐ AN `if` LEAVES ITS NEGATIVE FACT ON THE FALL-THROUGH EDGE and a `while`
# does not. The loop exits when its test is false, so the after-loop binding is
# narrowed by exactly the same reasoning -- the type channel just does not
# apply it there, and the value arrives at the return as the whole union.
#
# ⛔ NOT a wrong answer: the program does not compile. The MESSAGE named
# nothing at all until 2026-09-03 (a union bundle has no contract name, so it
# read "cannot adapt  return value"), which is what made the shape hard to
# recognise.
def f(flag: bool) -> str:
    v: "int | str" = 1 if flag else "a"
    while isinstance(v, int):
        v = "done"
    return v


print(f(True), f(False))
