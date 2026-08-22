# unboxed clone は「呼び出し側が有効な i64 レーンを渡す」ことを事前条件にする。
# 実行が必要な理由: 破れたときの症状が値ではなく発散 (無効レーンで clone に入る
# と body の分岐が全部反対側に倒れ、g は永久に自分を呼ぶ) なので、コンパイルが
# 通ることでは何も分からない。

# 引数のレーンが実行時に溢れる: clone に入ると符号が反転して見える
def sign(n: int) -> int:
    if n > 0:
        return 1
    return 2


# 引数がレーンから溢れる: clone に入ってはならない
def g(n: int) -> int:
    if n > 1000:
        return 1
    return g(n * 10000000000000000000)


# ループ内で桁溢れする蓄積: レーンは途中で無効になるが答えは bignum
def acc(times: int) -> int:
    total = 1
    i = 0
    while i < times:
        total = total * 3
        i = i + 1
    return total


# 再帰の戻り値側で溢れる
def pow3(n: int) -> int:
    if n <= 0:
        return 1
    return pow3(n - 1) * 3


print(sign(4000000000 * 4000000000))
print(sign(3))
print(g(2))
print(g(5000))
print(acc(5))
print(acc(70))
print(pow3(50))
print(pow3(5))
