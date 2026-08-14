# The multi-argument print renders each argument through the shared ladder and
# joins the pieces. When one argument has no rendering, falling through landed
# on the manifest print, whose arity is one, and the report was "builtin
# callable 'print' expects exactly one positional argument" -- the argument
# count is not the problem, and the count was the only thing that message
# mentioned.
#
# ⭐ The argument used to be `d.get("z")`, an unnarrowed `int | None`. A union
# renders by testing its tag now (cases/union_renders_by_tag.py), so it is no
# longer an example of anything unrenderable. A FUNCTION still is: it has no
# __str__ and no __repr__ this dispatch can resolve, which is exactly the
# condition this diagnostic reports.
def helper(x: int) -> int:
    return x


print(1, helper)
