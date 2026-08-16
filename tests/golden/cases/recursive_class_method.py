# What this pins: a class method that calls itself, and two that call each
# other. Both were refused -- "recursive class method call is not supported
# (total -> total): class method bodies are inlined at their call sites, so a
# call cycle has no base case to stop the expansion" -- which took the tree
# traversal with them, and a tree is what a recursive method is usually for.
#
# The method is already emitted as a `func.func` under its own symbol;
# inlining is what call sites DO, not the only thing they can do. Inside the
# cycle the call goes to that symbol instead, with the receiver as the leading
# positional -- exactly how a free function recurses.
#
# Why this needs to run rather than assert on a diagnostic: the two paths are
# not interchangeable. Inlining is what lets a base method's `self.who()` bind
# to the RECEIVER's real class at each site, and a symbol call fixes the callee
# at the defining class -- so taking the symbol everywhere would turn every
# overridden method into a base-class call. `Shape` below is the control for
# that: its `describe()` calls `self.name()`, `Circle` overrides `name`, and
# the answer has to be the override. Only running it says which callee was
# chosen.
#
# Every expected line is python3.14's.


# --- direct recursion, with the base case in an `if` -----------------------
class Countdown:
    def __init__(self, floor: int) -> None:
        self.floor = floor

    def step(self, n: int) -> int:
        if n <= 0:
            return self.floor
        return self.step(n - 1)


print(Countdown(0).step(3), Countdown(7).step(0), Countdown(-1).step(10))


# --- mutual recursion: no method calls itself -----------------------------
class Ping:
    def __init__(self, floor: int) -> None:
        self.floor = floor

    def ping(self, n: int) -> int:
        if n <= 0:
            return self.floor
        return self.pong(n)

    def pong(self, n: int) -> int:
        return self.ping(n - 1)


print(Ping(0).ping(3), Ping(5).ping(0))


# --- the tree, which is what this is for ----------------------------------
class Tree:
    def __init__(self, v: int) -> None:
        self.v = v
        self.kids: list["Tree"] = []

    def add(self, child: "Tree") -> "Tree":
        self.kids.append(child)
        return self

    def total(self) -> int:
        s = self.v
        for k in self.kids:
            s += k.total()
        return s

    def depth(self) -> int:
        best = 0
        for k in self.kids:
            d = k.depth()
            if d > best:
                best = d
        return best + 1

    def count(self) -> int:
        n = 1
        for k in self.kids:
            n += k.count()
        return n


root = Tree(1)
branch = Tree(2)
branch.add(Tree(4))
branch.add(Tree(5))
root.add(branch)
root.add(Tree(3))
print(root.total(), root.depth(), root.count())


# --- a recursion whose argument does the work -----------------------------
class Math:
    def fact(self, k: int) -> int:
        if k <= 1:
            return 1
        return k * self.fact(k - 1)

    def fib(self, k: int) -> int:
        if k < 2:
            return k
        return self.fib(k - 1) + self.fib(k - 2)


print(Math().fact(5), Math().fact(10), Math().fib(10))


# --- keyword arguments in the recursive call ------------------------------
# The symbol call packs positionals, and a recursive call names a signature
# this walk already has -- so a keyword that names a positional parameter is
# that parameter's slot. Carrying a setting down a recursion this way is
# ordinary, and refusing it sent the whole method back to the refusal above.
class Scaled:
    def __init__(self, v: int) -> None:
        self.v = v
        self.kids: list["Scaled"] = []

    def add(self, child: "Scaled") -> "Scaled":
        self.kids.append(child)
        return self

    def total(self, scale: int = 1, offset: int = 0) -> int:
        s = self.v * scale + offset
        for k in self.kids:
            s += k.total(scale=scale, offset=offset)
        return s


scaled = Scaled(1)
scaled.add(Scaled(2))
scaled.add(Scaled(3))
print(scaled.total(), scaled.total(scale=10))
print(scaled.total(scale=2, offset=1), scaled.total(2, offset=1))


class Step:
    def __init__(self, floor: int) -> None:
        self.floor = floor

    def down(self, n: int, step: int = 1) -> int:
        if n <= 0:
            return self.floor
        return 1 + self.down(n - step, step=step)


print(Step(0).down(6, step=2), Step(0).down(6, 3), Step(9).down(0))


# --- THE CONTROL: a non-recursive call still inlines, so an override wins --
class Shape:
    def name(self) -> str:
        return "shape"

    def describe(self) -> str:
        return "a " + self.name()


class Circle(Shape):
    def name(self) -> str:
        return "circle"


print(Shape().describe(), Circle().describe())
