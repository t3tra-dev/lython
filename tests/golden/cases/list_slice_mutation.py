a = [1, 2, 3, 4, 5]
a[1:3] = [9, 8, 7]
print(a)
a[:2] = []
print(a)
a[len(a):] = [100, 200]
print(a)
b = [0, 1, 2, 3, 4, 5, 6, 7]
b[::2] = [10, 20, 30, 40]
print(b)
b[7:1:-2] = [70, 50, 30]
print(b)
del b[1:7:2]
print(b)
del b[::-1]
print(b)
c = [1, 2, 3]
c[1:2] = c
print(c)
d = [1, 2, 3, 4]
del d[:]
print(d)
e = ["x", "y"]
e[-1:] = ["a", "b", "c"]
print(e)
n = 2
e[n:n] = ["ins"]
print(e)
