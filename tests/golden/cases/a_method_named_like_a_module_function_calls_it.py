# WHAT: a method reads the module function its own name spells -- `error` in
# `def error(self, msg): return error(msg)` -- as CPython does, where a class
# body's names are not visible from the methods in it.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the name was bound to the
# method itself (the binding that lets a nested def call itself), so the call
# went back into the method one argument short; which body answered is seen
# only in what it returns.
def error(msg: str) -> str:
    return "module error: " + msg


def tell(n: int) -> str:
    return "module tell %d" % n


class Tokenizer:
    def error(self, msg: str) -> str:
        return "[" + error(msg) + "]"

    def tell(self) -> str:
        return tell(7)

    def other(self) -> str:
        return self.error("x") + " " + self.tell()


t = Tokenizer()
print(t.error("bad escape"))
print(t.tell())
print(t.other())
