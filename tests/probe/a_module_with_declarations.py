# The imported half of `wb_what_an_imported_module_cannot_declare`. Every
# declaration here compiles when this file is the MAIN module, which is what
# makes the probe beside it a statement about the boundary.
from enum import Enum
from typing import Callable


class Color(Enum):
    RED = 1
    BLUE = 2


DOUBLE: "Callable[[int], int]" = lambda n: n + 1
