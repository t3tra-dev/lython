# probe: list フィールドを分岐内で再束縛し、素の束縛でローカルに取ったうえで
#   **そのローカルを使う** (フィールドではなくローカルを読む)。これが失敗するので、
#   list の素の束縛が通るのは「束縛が何も消費しない SSA コピーだから」であって
#   束縛という綴りに逃げ道があるわけではない。汚染は SSA コピーを越えて伝わる。
#   (`flow_ifone_w3list_plainbind.py` = 束縛のみ → 通る、が対照。
#    束縛を 2 段に鎖状につないでも通るところまで確認済み。)
# axes: acquire=call width=w3list op=rebind flow=ifone read=bind-then-use-local
# CLASSIFICATION: 3 loud 拒否 (MLIR verifier 失敗 = 最早境界での診断になっていない)
#   operand #0 does not dominate this use
# CPython 3.14 expects: 2


class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
n = len("ab")
if n == 2:
    x: list[int] = [1, 2]
    o.f = x
read: list[int] = o.f
print(len(read))
