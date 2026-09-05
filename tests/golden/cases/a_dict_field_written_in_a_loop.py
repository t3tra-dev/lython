# What: `obj.field[key] = value` inside a loop, where the field is a dict that
# was built non-empty. The evidence describing the field's contents was
# recorded where the field was BUILT, and a loop makes it a full iteration
# stale: the replace it drove released the value stored at construction on
# every trip, and reading the dict afterwards crashed. Runtime values, because
# the failure was a segfault with no diagnostic and only a read can show it.


class Bag:
    def __init__(self) -> None:
        self.table: dict[str, int] = {"a": 1}
        self.names: dict[str, str] = {"x": "one"}
        self.items: list[int] = [1]


def new_key() -> str:
    b = Bag()
    for i in range(3):
        b.table["c"] = i
    return str(sorted(b.table.items())) + " " + str(len(b.table))


def existing_key() -> str:
    b = Bag()
    for i in range(3):
        b.table["a"] = i
    return str(sorted(b.table.items())) + " " + str(len(b.table))


def growing() -> str:
    b = Bag()
    for i in range(3):
        b.table["k" + str(i)] = i
    return str(sorted(b.table.items())) + " " + str(len(b.table))


def string_values() -> str:
    b = Bag()
    for _ in range(3):
        b.names["y"] = "two"
    return str(sorted(b.names.items()))


def a_list_field_still_appends() -> str:
    b = Bag()
    for i in range(3):
        b.items.append(i)
    return str(b.items)


print(new_key())
print(existing_key())
print(growing())
print(string_values())
print(a_list_field_still_appends())
