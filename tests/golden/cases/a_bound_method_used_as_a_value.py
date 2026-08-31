# What: reading a method without calling it is the bound method, and only
# calling what came out of the container shows which receiver and which body
# it carried -- each answer here is different.
class Counter:
    def __init__(self, start: int) -> None:
        self.n = start

    def value(self) -> int:
        return self.n

    def doubled(self) -> int:
        return self.n * 2


one = Counter(1)
two = Counter(2)

readers = [one.value, two.value]
print([f() for f in readers])

both = [one.value, one.doubled]
print([f() for f in both])

table = {"a": one.value, "b": two.doubled}
print(table["a"](), table["b"]())


class Greeter:
    def hello(self) -> str:
        return "hi"


g = Greeter()
name = g.hello
print(name(), [f() for f in [g.hello, g.hello]])


class Adder:
    def plus(self, n: int) -> int:
        return n + 10


a = Adder()
print([f(5) for f in [a.plus, a.plus]])
