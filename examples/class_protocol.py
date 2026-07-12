class Seq:
    def __init__(self) -> None:
        self.items: list[int] = []

    def push(self, value: int) -> None:
        self.items.append(value)

    def __getitem__(self, index: int) -> int:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, value: int) -> bool:
        i = 0
        while i < len(self.items):
            if self.items[i] == value:
                return True
            i = i + 1
        return False


s = Seq()
s.push(10)
s.push(20)
s.push(30)
print(s[1])
print(s[2])
print(len(s))
print(20 in s)
print(99 in s)
print(99 not in s)

d: dict[str, int] = {}
d["a"] = 1
j = 0
while j < 3:
    d["k" + str(j)] = j
    j = j + 1
print("a" in d)
print("k2" in d)
print("zz" in d)
print("zz" not in d)
