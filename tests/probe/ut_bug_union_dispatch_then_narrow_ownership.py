# WHAT: A `Loud | StringIO` value used by a dispatched call (`str(w)`) and
#   then narrowed by isinstance for a field read. The ownership verifier
#   stops it: "released owned resource ... is used after release".
# KIND: バグ (bug) -- loud, from the ownership verifier
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   released owned resource from @_ZN8__main__19program_or_manifestE is used after release
# CPython 3.14 expects: < / 0
import io


class Loud:
    def __init__(self) -> None:
        self.n = 0

    def write(self, text: str) -> int:
        return len(text)


def program_or_manifest(flag: bool) -> Loud | io.StringIO:
    if flag:
        return Loud()
    return io.StringIO()


w = program_or_manifest(True)
print(str(w)[:1])
if isinstance(w, Loud):
    print(w.n)
else:
    print(repr(w.getvalue()))
