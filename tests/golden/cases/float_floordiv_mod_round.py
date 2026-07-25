# float.__floordiv__, __mod__ and __round__ were declared in the builtins
# manifest with no implementation behind them, and __truediv__ returned inf
# for a zero divisor instead of raising. Output is CPython 3.14's.

# Sign fixup: the remainder follows the divisor, the quotient floors.
print(7.5 // 2.0)
print(-7.5 // 2.0)
print(7.5 // -2.0)
print(-7.5 // -2.0)
print(7.5 % 2.0)
print(-7.5 % 2.0)
print(7.5 % -2.0)
print(-7.5 % -2.0)

# An exact division leaves a signed zero remainder that follows the divisor.
print(4.0 // 2.0)
print(-4.0 // 2.0)
print(4.0 % 2.0)
print(-4.0 % 2.0)
print(4.0 % -2.0)
print(-4.0 % -2.0)

# fmod stays exact where x - trunc(x/y)*y would not.
print(1e300 // 3.0)
print(1e16 % 3.0)

# The explicit method call is what `method_names` literally promises.
x: float = 7.5
y: float = 2.0
print(x.__floordiv__(y))
print(x.__mod__(y))

# round(x) narrows to int with ties to even; round(x, n) stays float.
print(round(2.5))
print(round(3.5))
print(round(-2.5))
print(round(3.7))
print(round(3.14159, 2))
print(round(1234.5678, -2))
print(round(2.5, 0))
print(x.__round__(1))

# int.__index__ is the identity on int.
i: int = 7
print(i.__index__())
print((-7).__index__())

# bool.__bool__ and the three bit operators.
b: bool = True
c: bool = False
print(b.__bool__())
print(b & c, b | c, b ^ c)
print(b.__and__(c), b.__or__(c), b.__xor__(c))
print(b & b, c ^ c)
