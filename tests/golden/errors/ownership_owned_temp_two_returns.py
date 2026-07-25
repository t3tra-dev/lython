# 所有ローカルの int 一時値が 2 つの return を跨ぐと release 配置が失敗する。
# 診断は loud (最も早い静的境界) だが、これは受理されるべきコードである。
#
# このファイルが固定するのは「拒否が loud であること」だけである。追跡単位が
# オブジェクト 1 個になるとレーン N 本すべてが全出口を dominate する要求が
# 消えるので綴りの一部は通るようになるが、配置アルゴリズムの path 感度不足は
# 独立成分として残る。どちらに転んでも黙って mis-execute してはならない。
def pick(a: int, b: int) -> int:
    t: int = a + b
    if a > 0:
        return t
    return t + 1


print(pick(1, 2))
