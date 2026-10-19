import types

Info = tuple[type[BaseException], BaseException, types.TracebackType]
Pair = tuple[int, str]
Names = list[str]
type Count = dict[str, int]


def first_name(info: Info) -> str:
    return info[0].__name__


def pair_text(p: Pair) -> str:
    return str(p[0]) + p[1]


class Recorder:
    def __init__(self) -> None:
        self.seen: Names = []
        self.counts: Count = {}

    def record(self, info: Info) -> Names:
        name = first_name(info)
        self.seen.append(name)
        self.counts[name] = self.counts.get(name, 0) + 1
        return self.seen
