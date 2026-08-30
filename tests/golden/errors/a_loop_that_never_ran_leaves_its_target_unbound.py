# What: the target of a zero-trip loop is bound by nothing, and CPython says so
# at the READ -- UnboundLocalError inside a function, naming the variable. The
# slot the binding uses has to raise rather than hand back a default, which is
# only observable by running it.
def first_index(n: int) -> int:
    for i in range(n):
        pass
    return i


print(first_index(0))
