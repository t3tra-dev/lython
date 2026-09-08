# OPEN, and found by the loop repair beside it: the recursive-descent parser
# below -- the grammar functions calling each other back through parentheses --
# is refused:
#
#     owned resource from @__ly_method$Parser$factor$25_4 result 0 is still
#     owned when a call to 'LyUnicode_FromBytes' may unwind out of the
#     function; the unwind path must release, transfer, or return it
#
# The owned value is `value = self.factor()`, live across `self.eat("*")` --
# whose string literal allocates and may unwind.
#
# MEASURED 2026-09-09, by shrinking it one feature at a time. EVERY smaller
# version compiles and runs:
#
#   two levels (`term` calls `number`), same loop and slice ....... correct
#   `while True` with an owned local and an unwinding call ........ correct
#   the same in a plain function rather than a method ............. correct
#   the whole grammar with one arm per loop ....................... correct
#   the same with a second `elif` arm in either loop .............. correct
#   this file, minus the raise / minus `/` / minus `-` / minus
#     `peek` ...................................................... all FAIL
#
# ⭐ SO THE TRIGGER IS SIZE, not any one feature: the four methods are inlined
# into each other until the budget stops them, and what does not survive that
# is the release placer's unwind cleanup for a local held across a call. Every
# subtraction that still fails leaves the same four-method cycle behind, and
# every version that passes has one fewer level in it.
#
# ⛔ The loop repair (cases/a_loop_whose_only_exit_is_a_return) is what made
# this reachable: before it the same program stopped at the fallthrough
# refusal, which is why a program this ordinary had never been seen to fail.
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
        if start == self.pos:
            raise ValueError("expected a number at " + str(self.pos))
        return int(self.text[start:self.pos])

    def factor(self) -> int:
        if self.eat("("):
            value = self.expr()
            if not self.eat(")"):
                raise ValueError("expected )")
            return value
        return self.number()

    def term(self) -> int:
        value = self.factor()
        while True:
            if self.eat("*"):
                value = value * self.factor()
            elif self.eat("/"):
                value = value // self.factor()
            else:
                return value

    def expr(self) -> int:
        value = self.term()
        while True:
            if self.eat("+"):
                value = value + self.term()
            elif self.eat("-"):
                value = value - self.term()
            else:
                return value


for source in ["1+2*3", "(1+2)*3", "10/3", "2*(3+4)-5"]:
    print(source, "=", Parser(source).expr())
try:
    Parser("1+").expr()
except ValueError as e:
    print("err:", e)
