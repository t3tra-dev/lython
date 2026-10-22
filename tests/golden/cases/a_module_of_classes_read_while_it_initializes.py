# The imported half of an_imported_module_initializes_top_to_bottom.


class Style:
    default_format = "%(message)s"

    def __init__(self, fmt: str | None = None) -> None:
        self.fmt = fmt or self.default_format


class Brace(Style):
    default_format = "{message}"


# Written below the classes, so it reads their attributes after they are set.
DEFAULT_STYLE = Style()
BRACE_STYLE = Brace()
NAMES: list[str] = [DEFAULT_STYLE.fmt, BRACE_STYLE.fmt]
