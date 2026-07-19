def check(name: str, truthy: bool) -> None:
    if truthy:
        print(name, "truthy")
    else:
        print(name, "falsy")
xs: list[int] = []
ys = [1]
check("empty-list", True if ys else False)
if xs:
    print("bad")
else:
    print("empty list falsy")
d: dict[str, int] = {}
if d:
    print("bad")
else:
    print("empty dict falsy")
t = (1, 2)
if t:
    print("tuple truthy")
s = ""
if not s:
    print("empty str falsy")
b = b"x"
if b:
    print("bytes truthy")
def opt(v: str | None) -> str:
    if v:
        return "have " + v
    return "none-or-empty"
print(opt(None))
print(opt(""))
print(opt("val"))
