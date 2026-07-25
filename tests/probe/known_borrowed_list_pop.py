# probe: borrowed list pop (shrink, no realloc)
# axes: acquire=param width=w3list op=pop flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   static type !py.contract<"builtins.list", [!py.contract<"builtins.int">]> does not provide manifest method 'pop'
# CPython 3.14 expects: 1

def drop(xs: list[int]) -> None:
    xs.pop()


xs: list[int] = [1, 2]
drop(xs)
print(len(xs))
