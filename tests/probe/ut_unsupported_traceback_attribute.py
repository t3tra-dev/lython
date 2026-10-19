# WHAT: `e.__traceback__`: the traceback is built from the runtime's frame
#   stack by the traceback module (`traceback.format_exception(e)`,
#   sys.exc_info()[2]).
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   '__traceback__' is not reachable as an attribute: the traceback is built from the runtime's frame stack by the traceback module, so write `traceback.format_exception(e)` for the formatted form or `traceback._current_tb()
# CPython 3.14 expects: True
try:
    raise ValueError("v")
except ValueError as e:
    print(e.__traceback__ is not None)
