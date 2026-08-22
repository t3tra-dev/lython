# while ループ内でループ変数からコンテナを作り、それを添字参照する形。
#
# 実行が必要な理由: 壊れ方が二段ある。コンパイルが通るかどうか (lowering 拒否)
# だけでは足りず、通ったあとに要素の一時参照が解放されているかは参照計数でしか
# 見えない。このファイルは leak stage にも登録してあり、そちらが後半を見る。
#
# p[0] はコンパイル時に畳み込まれてループ変数そのものになるので、要素の所有トー
# クンはループ変数の別名になる。所有権配置も検証もループ変数についての事実で答え
# てしまい、解放が置かれず、症状は 2 ブロック離れた `i < n` で報告されていた。


def list_only_counter(n: int) -> int:
    i = 0
    while i < n:
        p = [i]
        i = p[0] + 1
    return i


def tuple_only_counter(n: int) -> int:
    i = 0
    while i < n:
        p = (i, i)
        i = p[0] + 1
    return i


# ループ変数が 2 つ: 累算側は書き換えられ、カウンタ側の解放はループ出口にある
def list_accumulator(n: int) -> int:
    total = 0
    i = 0
    while i < n:
        p = [i]
        total = total + p[0]
        i = i + 1
    return total


def tuple_accumulator(n: int) -> int:
    total = 0
    i = 0
    while i < n:
        p = (i, i + 1)
        total = total + p[0] + p[1]
        i = i + 1
    return total


print(list_only_counter(5))
print(tuple_only_counter(5))
print(list_accumulator(5))
print(tuple_accumulator(5))
