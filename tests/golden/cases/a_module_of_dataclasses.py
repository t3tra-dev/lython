# Helper for an_imported_dataclass_renders_its_bare_name.
from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class Point:
    x: int
    y: int = 0

    def norm(self) -> int:
        return self.x * self.x + self.y * self.y


class Pair(NamedTuple):
    a: int
    b: str = "x"
