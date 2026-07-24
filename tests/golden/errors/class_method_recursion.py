class Countdown:
    def __init__(self, floor: int) -> None:
        self.floor = floor

    # Class method bodies are inlined at their call sites, and inlining emits
    # both arms of the `if`, so this cycle has no base case: it must be
    # rejected here rather than expanding until the emitter's stack overflows.
    def step(self, n: int) -> int:
        if n <= 0:
            return self.floor
        return self.step(n - 1)


print(Countdown(0).step(3))
