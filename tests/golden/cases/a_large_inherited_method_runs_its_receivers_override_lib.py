seen: list[str] = []


class Base:
    def __init__(self, name: str) -> None:
        self.name = name

    def hook(self, i: int) -> int:
        return i

    def report(self, label: str, sep: str = ",",
               limit: int | None = None) -> int:
        total = 0
        count = 0
        parts: list[str] = []
        for i in range(5):
            if limit is not None and i >= limit:
                break
            value = self.hook(i)
            total += value
            count += 1
            parts.append(str(value))
        seen.append(label + ":" + self.name)
        text = sep.join(parts)
        if total > 100:
            print(label, "big", text)
        elif total % 2 == 0:
            print(label, "even", text)
        else:
            print(label, "odd", text)
        print(label, "count", count, "seen", len(seen))
        return total


class Mid(Base):
    def hook(self, i: int) -> int:
        return i * 10
