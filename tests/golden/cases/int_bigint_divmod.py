"""Floor division / modulo / true division stay CPython-correct after the
operands leave the i64 fast path (digit-based long division)."""

f = 1
i = 1
while i <= 30:
    f = f * i
    i = i + 1
print(f)
print(f // 1000000)
print(f % 97)
print(f // f)
print(f % f)
print((0 - f) // 1000000)
print((0 - f) % 1000000)
print(f // (f - 1))
print(f % (f - 1))
print((0 - f) // (f - 1))
print((0 - f) % (f - 1))
print(f // (0 - f + 1))
print(f % (0 - f + 1))
big = 9223372036854775807 + 1
print(big // 2)
print(big % 2)
print(big // -2)
print(big % -2)
a = -9223372036854775807 - 1
print(a // -1)
print(a % -1)
print(f / f)
