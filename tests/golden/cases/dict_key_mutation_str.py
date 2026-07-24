# A str field is stored BOX-FRONTED, so a rebind after the instance became a
# dict key overwrites the box payload in place: the key box borrows the same
# stable box pointer, so the stored key observes the NEW string (CPython's
# stale-hash behaviour) instead of a freed payload. Before the box-fronted
# storage the inline (ptr, len) words went stale and the lookups compared
# freed memory (silent mis-eq) and double-released at teardown.
class K:
    def __init__(self, v: str) -> None:
        self.v = v

    def __hash__(self) -> int:
        return len(self.v) * 7

    def __eq__(self, other: "K") -> bool:
        return self.v == other.v

    def __repr__(self) -> str:
        return "K(" + repr(self.v) + ")"


k = K("a")
d = {}
d[k] = "x"
k.v = "b"
print(K("a") in d)
print(K("b") in d)
print(k in d)
print(len(d))
print(d[K("a")] if K("a") in d else "gone")

# A str field read escaping its owner: the load takes a reference, so the
# returned string survives the instance's release.
def name_of(value: K) -> str:
    return value.v


print(name_of(K("escaped")))
