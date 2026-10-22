# WHAT: an attribute a class neither declares nor inherits is read through
# its `__getattr__`, with the name the program wrote; a declared field is not;
# `__getattr__` raising AttributeError is what the read raises.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which names the instance
# holds is decided by what the program stored at run time.
#
# ⛔ It was refused ("__getattr__ is not called for a statically unresolved
# attribute"); argparse's Namespace is built on exactly this.


class Namespace:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}
        self.known = 1

    def set(self, name: str, value: object) -> None:
        self._values[name] = value

    def __getattr__(self, name: str) -> object:
        if name in self._values:
            return self._values[name]
        raise AttributeError(name)


class Upper:
    def __getattr__(self, name: str) -> str:
        return name.upper()


ns = Namespace()
ns.set("count", 3)
ns.set("name", "x")
print(ns.known, ns.count, ns.name)
count = ns.count
if isinstance(count, int):
    print(count + 1)
try:
    print(ns.missing)
except AttributeError as e:
    print("AttributeError", e)
print(Upper().shout + "!")
