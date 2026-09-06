# Helper for an_imported_hook_runs_when_its_module_is_read. The registry class
# and one subclass declared BESIDE it, so the hook has to fire for a class the
# importer never sees declared.
class Registry:
    seen: list[str] = []

    @classmethod
    def __init_subclass__(cls, tag: str = "none") -> None:
        Registry.seen.append(cls.__name__ + ":" + tag)


class Alpha(Registry, tag="a"):
    pass


class Middle(Registry, tag="m"):
    @classmethod
    def __init_subclass__(cls, tag: str = "none") -> None:
        Registry.seen.append("middle:" + cls.__name__)
