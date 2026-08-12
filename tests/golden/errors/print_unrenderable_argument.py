# The multi-argument print renders each argument through the shared ladder and
# joins the pieces. When one argument has no rendering, falling through landed
# on the manifest print, whose arity is one, and the report was "builtin
# callable 'print' expects exactly one positional argument" -- the argument
# count is not the problem, and the count was the only thing that message
# mentioned.
d: dict[str, int] = {"a": 1}
print(1, d.get("z"))
