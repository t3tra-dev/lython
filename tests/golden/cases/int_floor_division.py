"""Floor division and modulo follow CPython (floor) semantics for every
sign combination, including the i64 boundary."""

print(-7 // 2)
print(-7 % 2)
print(7 // -2)
print(7 % -2)
print(7 // 2)
print(7 % 2)
print(-7 // -2)
print(-7 % -2)
print(-8 // 2)
print(-8 % 2)
print(0 // 5)
print(0 % 5)
print(-1 // 9223372036854775807)
print(-1 % 9223372036854775807)
