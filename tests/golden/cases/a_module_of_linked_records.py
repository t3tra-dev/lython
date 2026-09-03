# Helper for an_imported_class_whose_field_arrives_later.
LIMIT = 3


class Record:
    kind = "record"

    def __init__(self, name: str) -> None:
        self.name = name
        self.parent = None

    def label(self) -> str:
        return self.kind + ":" + self.name

    def chain(self) -> str:
        if self.parent is None:
            return self.label()
        return self.parent.chain() + ">" + self.label()

    def attach(self, other: "Record") -> None:
        other.parent = self


class User(Record):
    kind = "user"

    def label(self) -> str:
        return super().label().upper()


class Store:
    capacity = LIMIT

    def __init__(self) -> None:
        self.items: list[Record] = []

    def add(self, r: Record) -> bool:
        if len(self.items) >= Store.capacity:
            return False
        self.items.append(r)
        return True
