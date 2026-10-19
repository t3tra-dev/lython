# A field declared `object` holds whatever is put in it. The store was
# refused ("a type-erased object value cannot be stored in field") for a
# slot layout of sixteen words where an erased handle read back wrong across
# a call -- silently for a str, float, list or dict. A slot is the entity
# alone now, and the read rebuilds the box from the entity's own header, so
# the class it answers with is the one stored LAST, through whatever name.
# Why run it: each payload has to come back as itself across a call, and a
# None stored through the slot a reused module name lives in has to read
# back as None rather than as the instance the first store left.


class Inner:
    def __init__(self, n: int) -> None:
        self.n = n

    def __repr__(self) -> str:
        return "Inner(%d)" % self.n


class Holder:
    def __init__(self, v: object) -> None:
        self.f: object = v


def show(h: Holder) -> str:
    return repr(h.f)


def replace(h: Holder, v: object) -> None:
    h.f = v


class Marker:
    pass


sentinel = Marker()
values: list[object] = ["text", 2.5, 7, 2**70, [1, 2], {"k": 3}, Inner(4), None, True, sentinel]
holders = [Holder(v) for v in values]
for h in holders:
    print(type(h.f).__name__, show(h) if h.f is not sentinel else "<sentinel>", h.f is None)
replace(holders[0], "other")
print(show(holders[0]), holders[-1].f is sentinel, holders[0].f == "other")
h = Holder(Inner(9))
print(h.f, h.f is None)
h.f = None
print(h.f, h.f is None)
h.f = [3]
print(h.f)
