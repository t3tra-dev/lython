# WHAT: `BaseException.__str__(e)` asks for the BASE method, which formats
#   the stored args ('stored'); here it reaches the subclass override
#   ('fancy text').
# KIND: バグ (bug) -- silent: prints the wrong text
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- `BaseException.__str__(x)` and `__repr__` are CPython's
#   BaseException_str/repr over `x.args`; a call naming a builtin class is not
#   virtual (`int.__repr__(True)` is "1").
# CLASSIFICATION @ lang-gaps 8639cfed + the fix: 1 正しい
# CPython 3.14 expects: stored fancy text
class Fancy(Exception):
    def __str__(self) -> str:
        return "fancy text"


e = Fancy("stored")
print(BaseException.__str__(e), str(e))
