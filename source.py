# hello, world!
print("Hello, world!")

# 非ASCII文字列
print("multibite character: あいうえお, 🐍")


# フィボナッチ数列を計算する
def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


print(str(fib(10)))
