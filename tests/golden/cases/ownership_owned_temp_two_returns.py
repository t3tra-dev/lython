# 所有ローカルの int 一時値が 2 つの return を跨ぐ形。かつては release 配置が
# 失敗し loud に拒否されていた (errors/ 側にあった) が、配置が region op の
# 名前を見るようになって受理される。このファイルが固定するのは CPython と同じ
# 値を返すこと。
#
# Why execution: 拒否から受理に変わったので、固定すべきものが診断ではなく値に
# なった。ここが再び拒否になるなら配置の後退であり、違う値を返すなら所有権の
# 後退である。どちらも実行しないと見えない。
def pick(a: int, b: int) -> int:
    t: int = a + b
    if a > 0:
        return t
    return t + 1


print(pick(1, 2))
