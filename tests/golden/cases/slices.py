s = "hello world"
print(s[1:4])
print(s[:5])
print(s[6:])
print(s[::2])
print(s[::-1])
print(s[-5:-1])
print(s[10:2:-2])
u = "héllo wörld"
print(u[1:4])
print(u[::-1])
print(len(u[2:]))
a = [1, 2, 3, 4, 5]
print(a[1:3])
print(a[::-1])
print(a[::2])
print(a[-2:])
print(len(a[7:]))
b2 = a[1:]
print(b2[0])
t = (1, 2, 3, 4)
print(t[1:3])
print(t[::-1])
bs = b"abcdef"
print(bs[1:4])
print(bs[::-1])
n = 2
print(a[:n])
print(a[n:][0])
words = ["x", "y", "z", "w"]
for w in words[1:3]:
    print(w)
print("abc"[2:0])
