# `while True:` with no break cannot reach the statement after it, and the
# analysis that decides whether a function falls through did not model it:
#
#     def term(self) -> int:
#         value = self.factor()
#         while True:
#             if self.eat("*"):
#                 value = value * self.factor()
#             else:
#                 return value
#     # this function can reach its end without returning, and its declared
#     # result builtins.int cannot hold the None a fallthrough returns
#
# -- the shape every recursive-descent parser's `term()` and `expr()` are
# written in, refused for a fallthrough that cannot happen.
#
# Why execution: the answers are the loop's, and a wrong model of the exit
# would either refuse the program (it did) or put a raise where the return is.
# The `else`-arm and both-arms-return spellings are here because they take
# different paths through the completion walk.
#
# ⭐ The IR says it too now: a constant-true header branches unconditionally to
# the body, so the after-block has only the `break` edges it should have. That
# is what the INLINED method's check reads -- it runs after the IR is built,
# and a method with this shape was refused by that copy of the question even
# once the function-level one was fixed.
#
# ⛔ An `else` on such a loop keeps the conservative answer: it is dead code in
# CPython too, and reading it as unreachable is the direction that puts a raise
# at a function's end.
class Parser:
    def __init__(self, text: str) -> None:
        self.text: str = text
        self.pos: int = 0

    def peek(self) -> str:
        if self.pos >= len(self.text):
            return ""
        return self.text[self.pos]

    def eat(self, ch: str) -> bool:
        if self.peek() == ch:
            self.pos += 1
            return True
        return False

    def number(self) -> int:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        return int(self.text[start:self.pos])

    def term(self) -> int:
        value = self.number()
        while True:
            if self.eat("*"):
                value = value * self.number()
            elif self.eat("/"):
                value = value // self.number()
            else:
                return value


def first_over(xs: "list[int]", limit: int) -> int:
    i = 0
    while True:
        if i >= len(xs):
            return -1
        if xs[i] > limit:
            return xs[i]
        i += 1


def both_arms(n: int) -> int:
    while True:
        if n > 3:
            return n
        return -n


def nested(n: int) -> int:
    while True:
        while True:
            if n > 5:
                return n
            n += 1


def with_a_break(n: int) -> int:
    while True:
        if n > 3:
            break
        n += 1
    return n


def growing(text: str) -> str:
    while True:
        if len(text) >= 4:
            return text
        text = text + "x"


def main() -> None:
    print(Parser("2*3*4").term(), Parser("12/3").term(), Parser("7").term())
    print(first_over([1, 5, 9], 4), first_over([1], 4))
    print(both_arms(9), both_arms(1))
    print(nested(1), with_a_break(1), growing("a"))


main()
