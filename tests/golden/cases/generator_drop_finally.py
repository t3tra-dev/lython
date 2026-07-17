# RAII on implicit drop (rfc/stdlib-semantics.md R3): a generator abandoned
# mid-suspension is finalized when its refcount reaches zero — the drop
# dispatch runs close semantics (GeneratorExit into the body, so the finally
# executes) and releases the frame's absorbed values (the list carried
# across the yields). CPython runs the same finalization from gen.__del__.
def gen():
    items: list[int] = []
    items.append(1)
    try:
        yield 1
        yield 2
        items.append(2)
    finally:
        print("finally")


def use() -> int:
    g = gen()
    value = next(g)
    return value


print(use())
print("after")
