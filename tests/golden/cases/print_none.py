# Why execution: what is printed IS the fix. None has no lanes, so there is no
# receiver to hand a `__repr__` and nothing to ask -- every None renders as the
# same four bytes. print() asked for a header anyway and was refused with
# "types.NoneType runtime object has no physical header value", which named the
# ABI rather than the answer. A compile check would pass on a fold that emitted
# the wrong string.

def returns_none() -> None:
    return None


def main() -> None:
    print(None)
    value = None
    print(value)
    print(returns_none())
    print(str(None))


main()
