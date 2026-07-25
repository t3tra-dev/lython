# 呼び出しから受け取ったインスタンスの list フィールドを再代入する。フィールドの
# 物理レーンは代入で差し替わるが、そのインスタンスの解放は生成時のレーンを指し
# 続けるので、代入側も同じ値を解放すると置き換えられた list が二重解放になる
# (負荷やアロケータ状態次第で SIGABRT / use-after-free になり、単体実行では
# たまたま通る)。読み出した長さと要素が毎回正しく、正常終了しなければならない。


class Node:
    def __init__(self, kids: list["Node"]) -> None:
        self._kids: list["Node"] = kids


def leaf() -> Node:
    empty: list[Node] = []
    return Node(empty)


# 1 回の再代入。
one = leaf()
first: list[Node] = [leaf()]
one._kids = first
print(len(one._kids))

# 分岐をまたいで 2 回再代入する (2 回目は分岐の合流後、別ブロックで起きる)。
# 再代入を分岐の内側やループ本体で行う綴りは別の欠陥 (MLIR dominance 失敗) に
# あたるので、ここでは扱わない。
two = leaf()
a: list[Node] = [leaf()]
two._kids = a
if len(two._kids) > 0:
    print("nonempty")
b: list[Node] = [leaf(), leaf()]
two._kids = b
print(len(two._kids))

# 空 list への差し戻し。
three = leaf()
c: list[Node] = [leaf(), leaf(), leaf()]
three._kids = c
print(len(three._kids))
d: list[Node] = []
three._kids = d
print(len(three._kids))

# 関数内で生成したインスタンスでも同じ (こちらは元から通っていた経路)。
def local_rebind() -> int:
    empty: list[Node] = []
    node = Node(empty)
    kids: list[Node] = [leaf(), leaf()]
    node._kids = kids
    return len(node._kids)


print(local_rebind())
