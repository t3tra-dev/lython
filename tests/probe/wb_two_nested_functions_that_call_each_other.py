# Mutual recursion between two functions nested in the same body is refused:
#
#   emit error: unresolved name 'od'
#
# The self-recursive spelling works (golden: a_nested_function_that_calls_
# itself), and so does the TOP-LEVEL mutual pair -- `def ev` / `def od` at
# module scope compiles and answers.
#
# ⭐ A DEF STATEMENT BINDS ITS NAME WHEN IT EXECUTES, and the sibling below it
# has not executed yet. The module walk gets around that by DECLARING every
# top-level def before any body is emitted; a nested body is emitted in one
# pass, in statement order, so `ev` reading `od` reads a name nothing has
# bound. The self-reference this compiler now makes is available because the
# function's own symbol is known at the point its body is emitted -- a
# sibling's is not, because the sibling has not been given one.
#
# ⛔ Not fixable by binding the name to the sibling's eventual symbol: the
# sibling's closure arguments are values of the ENCLOSING frame, and at the
# point `ev` is emitted the sibling's captures have not been evaluated -- the
# reference would carry the wrong ones. The top-level pair has no captures,
# which is why the declare-first pass is enough there.
def mutual(n: int) -> bool:
    def ev(k: int) -> bool:
        if k == 0:
            return True
        return od(k - 1)

    def od(k: int) -> bool:
        if k == 0:
            return False
        return ev(k - 1)

    return ev(n)


print(mutual(4), mutual(5))
