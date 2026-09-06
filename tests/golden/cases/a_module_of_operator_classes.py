# Helper for an_imported_class_is_not_a_manifest_type. Everything here is a
# class this compiler declares; the point of the other file is that crossing
# the module boundary must not turn any of them into a manifest type.
class Bag:
    def __init__(self) -> None:
        self.xs = [1, 2, 3]

    def __len__(self) -> int:
        return 3

    def __getitem__(self, i: int) -> int:
        return self.xs[i]


class Big:
    def __init__(self, n: int) -> None:
        self.n = n

    def __gt__(self, other: "Big") -> bool:
        return self.n > other.n


class Num:
    def __init__(self, n: int) -> None:
        self.n = n

    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return 0

    def __radd__(self, other: int) -> int:
        return other + 100


class Slot:
    def __init__(self, n: int) -> None:
        self.n = n

    def __index__(self) -> int:
        return self.n


class MyErr(Exception):
    pass
