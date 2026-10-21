# WHAT: a class of another module held in a `type[...]` parameter, in an
# Optional one narrowed by a default, and in a module global, each called with
# keyword arguments; and that module's `assert` over its constants.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the call reached the type
# object instead of the constructor, which the lowering could not bind a
# keyword to ("kw names lowering is not keyword-aware yet"); the arguments the
# instance holds are what show the keywords reached their parameters. The
# imported `assert` was refused as a module-level statement.
import a_class_value_of_another_module_called_with_keywords_lib as lib


def build(cls: type[lib.Runner]) -> lib.Runner:
    return cls(verbosity=4)


def build_default(cls: type[lib.Runner] | None = None) -> lib.Runner:
    if cls is None:
        cls = lib.Runner
    return cls(verbosity=2, durations=None)


k = lib.Runner
r = k(durations=3)
print(build(lib.Runner).verbosity, build_default().verbosity, r.verbosity, r.durations)
print(lib.margin())
