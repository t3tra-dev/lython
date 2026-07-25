# clone 内の比較は i1 を返すので「raw が真値でない」を後段へ運べない。運べないまま
# 分岐すると、その先の literal return が valid=true を主張して誤答が確定する。


def branch(c: int) -> int:
    if c >= 48:
        return c - 48
    return -1


# ord() 由来の値は箱側が真値を持ち raw lane は未確定
def code_at(text: str, at: int) -> int:
    return ord(text[at])


print(code_at("41", 0))
print(branch(code_at("41", 0)))
print(branch(ord("4")))
print(branch(code_at("41", 1)))

# len() 由来 (元から通っていた経路の退行防止)
print(branch(len("0" * 50)))


# clone 内の乗算が i64 を溢れると、その後の比較は raw では答えられない
def overflowed(n: int) -> int:
    m = n * n * n * n * n
    if m > 10:
        return 1
    return 0


print(overflowed(2))
print(overflowed(1000000))


# 溢れた raw を負値と誤認しない (溢れは符号も壊す)
def overflowed_sign(n: int) -> int:
    m = n * n * n
    if m < 0:
        return -1
    return 1


print(overflowed_sign(3))
print(overflowed_sign(3000000))

# 溢れた値を引数として別の clone 持ち関数へ渡す
print(branch(overflowed_sign(3000000)))
