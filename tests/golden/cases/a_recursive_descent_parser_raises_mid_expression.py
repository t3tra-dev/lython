# Why execution: the raise paths are the subject -- each malformed input
# raises from inside a loop that holds the value parsed so far, and only
# running shows the value was released there (the leak gate) and the right
# message reached the handler.
#
# A recursive-descent parser: the grammar methods are inlined into each other,
# so a `raise` in an inlined `factor` sits on an edge of a loop whose carried
# `value` is forwarded on the OTHER edge of the same branch. That edge got no
# release whenever the loop body also released the old value
# (`value = value * self.factor()`), and the program was refused: "owned
# resource ... is still owned when 'LyValueError_Raise' unwinds".
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


for source in ["1+2*3", "(1+2)*3", "10/3", "2*(3+4)-5", "((7))"]:
    print(source, "=", Parser(source).expr())
for bad in ["1+", "(1+2", "2*(3+", "((1)", "1+(2*", "*"]:
    try:
        Parser(bad).expr()
    except ValueError as e:
        print(repr(bad), "->", e)
