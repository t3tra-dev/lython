# クラスフィールドの list を変異させる 2 つの綴り。フィールドは header 間接の箱
# 越しに読まれるので、直接 append しても、いったんローカルに束ねて変異して書き戻し
# ても、同じ 1 本の list を指す。len と要素の両方が一致しなければならない。


class Stack:
    def __init__(self) -> None:
        self.items: list[int] = []
        self.names: list[str] = []

    # 直接変異
    def push(self, item: int) -> None:
        self.items.append(item)

    # read-mutate-rebind (同一ブロック内)
    def push_rebind(self, item: int) -> None:
        items = self.items
        items.append(item)
        self.items = items

    def push_name(self, name: str) -> None:
        self.names.append(name)

    def size(self) -> int:
        return len(self.items)


s = Stack()
s.push(1)
s.push_rebind(2)
s.push(3)
s.push_rebind(4)
print(s.size())
print(s.items)

# 二つ目の参照型フィールドも独立に変異する
s.push_name("a")
s.push_name("b")
print(s.names)
print(len(s.names))

# 外側から直接変異しても同じ list
s.items.append(99)
print(s.items)
print(s.size())

# 別インスタンスはフィールドを共有しない
t = Stack()
t.push(7)
print(t.items)
print(s.items)


# メソッドを介さず、直線ブロックで繰り返し変異する
u = Stack()
u.push(0)
u.items.append(1)
u.push_rebind(2)
u.items.append(3)
print(u.items)
print(u.size())
