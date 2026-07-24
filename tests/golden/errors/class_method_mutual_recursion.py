class Ping:
    def __init__(self, floor: int) -> None:
        self.floor = floor

    # No method calls itself directly; the cycle is ping -> pong -> ping.
    def ping(self, n: int) -> int:
        if n <= 0:
            return self.floor
        return self.pong(n)

    def pong(self, n: int) -> int:
        return self.ping(n - 1)


print(Ping(0).ping(3))
