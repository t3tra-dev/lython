# Helper for a_base_imported_by_name_is_still_a_hierarchy. The subclass lives
# in the OTHER file on purpose: what is being pinned is that a base reached by
# `from ... import Shape` is the same hierarchy as one reached by
# `shapes.Shape`, and only a base declared here can be spelled both ways.
class Shape:
    kind: str = "shape"

    def __init__(self, size: int) -> None:
        self.size = size

    def name(self) -> str:
        return "shape"

    def describe(self) -> str:
        return self.name() + str(self.size)

    def __len__(self) -> int:
        return 1

    @property
    def area(self) -> int:
        return 0

    @staticmethod
    def sides() -> int:
        return 0
