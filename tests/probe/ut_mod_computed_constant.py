# WHAT: A module with constants computed from other constants. Run alone it
#   prints nothing.
# KIND: helper module of ut_unsupported_computed_constant_of_imported_module
_MAX = 80
_PLACE = 12
_DIFF = _MAX - (_PLACE + 5)


def show() -> str:
    return str(_DIFF)
