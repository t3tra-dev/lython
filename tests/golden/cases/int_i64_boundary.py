"""Promotion boundary: arithmetic that crosses INT64_MAX / INT64_MIN must
promote to the digit form instead of wrapping."""

m = 9223372036854775807
print(m)
print(m + 1)
print(m + m)
print(m * m)
print(m * m * m)
print((m + 1) - 1)
print((m + 1) - (m + 1))
n = -9223372036854775807 - 1
print(n)
print(n - 1)
print(n + n)
print(n * n)
print(0 - n)
print(n * -1)
print((n + n) // 2 == n)
print(m + 1 > m)
print(n - 1 < n)
print(n < m)
print((m + 1) * 0)
print((m + 1) % (m + 1))
