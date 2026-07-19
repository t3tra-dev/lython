class Celsius:
    def __init__(self, temp: float) -> None:
        self._temp = temp

    @property
    def temp(self) -> float:
        return self._temp

    @temp.setter
    def temp(self, value: float) -> None:
        if value < 0.0:
            print("clamping to zero")
            value = 0.0
        self._temp = value

    @property
    def fahrenheit(self) -> float:
        return self._temp * 9.0 / 5.0 + 32.0


c = Celsius(25.0)
print(c.temp)
print(c.fahrenheit)
c.temp = 30.0
print(c.temp)
c.temp = -12.5
print(c.temp)
print(c.fahrenheit)
