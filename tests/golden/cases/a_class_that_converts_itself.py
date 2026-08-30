# What: the one-argument builtins whose whole job is to call a dunder, on a
# class that provides one -- and `round`, which is the one with a second
# argument. Running it is what shows the call reached the class's method: the
# manifest overloads these names also have answer these arguments, so a miss
# is a different value rather than a failure.
class Reading:
    def __init__(self, whole: int) -> None:
        self.whole = whole

    def __round__(self, digits: int = 0) -> int:
        return self.whole * 100 + digits

    def __bytes__(self) -> bytes:
        return b"raw"

    def __complex__(self) -> complex:
        return complex(self.whole, 1)

    def __abs__(self) -> int:
        return self.whole


reading = Reading(4)
print(round(reading), round(reading, 2))
print(bytes(reading), complex(reading), abs(reading))
print(round(2.567, 2), round(2.5), abs(-3))
