# int 引数・int 戻り値の関数は unboxed な (i64, i1) clone を持ち、呼び出し側は
# clone を投機実行して valid=false なら箱経路で受け直す。その受け直しは本体の
# 再実行なので、副作用は観測回数が変わってはならない。


def counted_while(n: int) -> int:
    i = 0
    while i < n:
        i = i + 1
    print("while-body")
    return i


# 別の関数から呼ぶ: 呼び出し側の return は py.call の結果なので clone を持つが、
# その clone は箱経路の callee を呼ぶので valid=false を返し得る
def call_while(n: int) -> int:
    return counted_while(n)


def counted_for(n: int) -> int:
    total = 0
    for k in range(n):
        total = total + k
    print("for-body")
    return total


def call_for(n: int) -> int:
    return counted_for(n)


# 二段重ね: 中間の関数も clone を持つ
def call_call_while(n: int) -> int:
    return call_while(n)


print(call_while(3))
print(call_for(4))
print(call_call_while(2))

# モジュール直下からの直接呼び出しは元から 1 回だった (退行防止)
print(counted_while(3))

# 副作用がグローバル状態の場合も 1 回でなければならない
hits: int = 0


def bump(n: int) -> int:
    global hits
    hits = hits + 1
    i = 0
    while i < n:
        i = i + 1
    return i


def call_bump(n: int) -> int:
    return bump(n)


print(call_bump(5))
print(hits)
