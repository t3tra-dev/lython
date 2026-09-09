# OPEN, and a feature boundary rather than a mechanism. A record written as a
# dict with STRING LITERAL keys has a different type per key, and the dict
# contract carries one value type -- so reading a key back gives the union of
# all of them:
#
#     TASKS.append({"title": title, "done": done})
#     ...
#     out.append(t["title"])
#     # !py.union<builtins.bool, builtins.str> does not provide '__add__'
#
# which is how a program with no classes in it carries a record, and how JSON
# arrives.
#
# MEASURED 2026-09-09, RelWithDebInfo:
#
#   `{"title": t, "done": d}` then `d["title"] + "!"` ......... refused
#   `{"items": [], "head": 0}` then `q["head"] + 1` ........... refused
#   the same fields on a small CLASS ......................... correct
#   a dict whose values are all one type ..................... correct
#   a TUPLE record, read positionally ........................ correct
#
# ⭐ THE KEYS ARE LITERALS, WHICH IS WHAT MAKES IT DECIDABLE. Every read in
# these programs is `d["title"]`, never `d[k]`, so a dict literal with literal
# keys could carry a per-key type the way a positional tuple carries a per-
# position one -- `tuple[A, B]` is already spelled that way in this compiler.
#
# ⛔ Not built here: it is a new shape in the type system (a record type), not
# a walk that failed to look somewhere, and every reader of `builtins.dict`
# would have to be told what to do when the key is not a literal. The
# workaround is a class, which this compiler handles well.
#
# ⛔ The failure direction is a refusal wherever the value is decoded, never a
# wrong field.
def make(title, done):
    return {"title": title, "done": done}


row = make("write", False)
print(row["title"] + "!")
