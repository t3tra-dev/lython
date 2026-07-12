class User:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return "User(name=" + self.name + ")"


e: list[int] = []
print(e)
xs: list[str] = []
xs.append("a")
xs.append("b'c")
print(xs)
ys: list[int] = []
i = 0
while i < 3:
    ys.append(i * 10)
    i = i + 1
print(ys)
d: dict[str, int] = {}
d["alpha"] = 1
d["beta"] = 2
print(d)
t = (1,)
print(t)
print((2.5, "x", 7))
nested = [[1, 2], [3]]
print(nested)
u = User("Zoe")
print(u)
print(repr(u))
us: list[User] = []
j = 0
while j < 2:
    us.append(User("u" + str(j)))
    j = j + 1
print(us)
print(us[1])
for u in us:
    print(u.name)
