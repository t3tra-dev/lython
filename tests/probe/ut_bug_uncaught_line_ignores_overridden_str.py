# WHAT: An exception that overrides __str__ escapes the program. CPython's
#   last traceback line is `Fancy: fancy text` -- str(exc); here it is the
#   message the constructor stored, `Fancy: stored`. unittest prints errors
#   through traceback.format_exception_only, which is right; the runtime's
#   own printer is not.
# KIND: バグ (bug) -- silent: the uncaught summary prints the stored message
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# differential: skip the difference is on stderr (the uncaught summary)
# CLASSIFICATION @ lang-gaps fa5c535a: 2 silent 誤実行
#   stderr: CPython 'Fancy: fancy text' vs lyc 'Fancy: stored' (stdout and exit status agree)
# CPython 3.14 expects: (例外を送出)
class Fancy(Exception):
    def __str__(self) -> str:
        return "fancy text"


raise Fancy("stored")
