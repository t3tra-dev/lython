# WHAT: `BaseException.__str__(e)` asks for the BASE method, which formats
#   the stored args ('stored'); here it reaches the subclass override
#   ('fancy text').
# KIND: バグ (bug) -- silent: prints the wrong text
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# differential: expect-wrong an explicit base __str__ call is dispatched to the override
# CLASSIFICATION @ lang-gaps fa5c535a: 2 silent 誤実行
#   cpython='stored fancy text\n' lyc='fancy text fancy text\n'
# CPython 3.14 expects: stored fancy text
class Fancy(Exception):
    def __str__(self) -> str:
        return "fancy text"


e = Fancy("stored")
print(BaseException.__str__(e), str(e))
