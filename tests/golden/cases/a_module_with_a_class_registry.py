# Helper for an_imported_class_attribute_is_storage. A registry class is the
# ordinary shape a library uses one for: a container everything appends to and
# a counter everything bumps, both class attributes.
class Registry:
    seen: list[str] = []
    count: int = 0
    label: str = "reg"

    @classmethod
    def record(cls, name: str) -> int:
        Registry.seen.append(name)
        Registry.count += 1
        return Registry.count
