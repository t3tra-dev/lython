class Tagged[T]:
    label = "tag"

    def __init__(self, value: T) -> None:
        self.value = value


# A monomorphized generic has one class per instantiation and no class object
# of its own, so the bare name cannot be used as one.
print(Tagged.label)
