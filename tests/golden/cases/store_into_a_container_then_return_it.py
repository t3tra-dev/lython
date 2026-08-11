# Why execution: memoization did not compile -- "owned resource from
# @LyLong_FromI64 result 0 is returned with 1 additional retained ownership
# token(s)". The reference the container holds is real, so these pin the
# returned values AND sit in the leak gate: subtracting a token that was not
# actually parked would compile and then free what the container still holds.


def memo_fib(n: int, memo: dict[int, int]) -> int:
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    v = memo_fib(n - 1, memo) + memo_fib(n - 2, memo)
    memo[n] = v
    return v


def store_then_return(n: int, memo: dict[int, int]) -> int:
    v = n * 2
    memo[n] = v
    return v


def store_str_then_return(k: str, m: dict[str, str]) -> str:
    v = k + "!"
    m[k] = v
    return v


def append_then_return(xs: list[int], n: int) -> int:
    v = n * 3
    xs.append(v)
    return v


def store_but_return_other(n: int, memo: dict[int, int]) -> int:
    v = n * 2
    memo[n] = v
    return n


def main() -> None:
    m: dict[int, int] = {}
    print(memo_fib(30, m))
    print(store_then_return(3, m))
    s: dict[str, str] = {}
    print(store_str_then_return("a", s), s["a"])
    xs: list[int] = []
    print(append_then_return(xs, 4), xs)
    print(store_but_return_other(5, m), m[5])


main()
