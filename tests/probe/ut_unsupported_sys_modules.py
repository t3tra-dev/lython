# WHAT: `sys.modules`: modules are bound when the program is compiled, so
#   there is no run-time table. unittest's module fixtures (setUpModule) and
#   its loader look modules up there.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   module 'sys' has no attribute 'modules' that resolves statically
# CPython 3.14 expects: True
import sys

print("__main__" in sys.modules)
