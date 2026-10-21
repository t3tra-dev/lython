# WHAT: `result.errors.append(result.render(...))` in a loop, where `render`
# is inlined and has a loop of its own, on a result built in the same function;
# then 40 appends, past the list's first capacity.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the field was read in one
# block and appended to in another, which the lowering refused ("list.append
# on a field or borrowed list is not supported inside a branch or loop body");
# the list's contents after the appends are the answer.
class Result:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def render(self, err: list[str]) -> str:
        lines: list[str] = []
        for line in err:
            lines.append(line)
        return ''.join(lines)


def run() -> Result:
    result = Result()
    for n in range(2):
        result.errors.append(result.render(["v", str(n)]))
    for k in range(40):
        result.errors.append(result.render(["x", str(k)]))
    return result


done = run()
print(done.errors[:3], len(done.errors), done.errors[-1])
