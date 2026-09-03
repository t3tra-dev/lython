# OPEN. A field initialized from the class's OWN class attribute types as
# `object`, so its reads carry nothing:
#
#     class Config:
#         DEFAULTS = {"a": 1}
#         def __init__(self) -> None:
#             self.values = dict(Config.DEFAULTS)
#         def get(self, k: str) -> int:
#             return self.values[k]
#     # builtins.object does not provide manifest method '__getitem__'
#
# The same line reading ANOTHER class's attribute compiles, and so does the same
# field with an annotation, or with the dict as a module global. So the defect
# is the SELF reference during the class's own emission.
#
# ⛔ MEASURED AND DROPPED: collecting the class attributes BEFORE the fields and
# seeding `classStaticAttrBindings` with them. It fixed nothing, because the
# class's own CONTRACT is not registered either at that point -- `Config` does
# not resolve as a type, so `Config.DEFAULTS` cannot be looked up whatever the
# attribute table says. Registering the contract that early is the real
# mechanism, and it is a reordering with the field list not yet known.
#
# ⛔ THE WORKAROUND IS THE ANNOTATION, and it is the same one the whole
# pre-pass family has: `self.values: dict[str, int] = dict(Config.DEFAULTS)`.
#
# Measured 2026-09-04.
class Config:
    DEFAULTS = {"a": 1}

    def __init__(self) -> None:
        self.values = dict(Config.DEFAULTS)

    def get(self, k: str) -> int:
        return self.values[k]


print(Config().get("a"))
