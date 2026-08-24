# payload box が運べるハンドルは 3 本。クラスのインスタンスはフィールドを何本
# 持っていても 1 本 (フィールドは body にあり、box にはインスタンスのハンドルが
# 入る) なので、ここに到達するのは union だけになった。union の storage は tag +
# 各メンバーのレーンで、メンバー同士は幅が違うから 1 つの box では前に出せない。
#
# int | str = tag + int 1 本 + str 1 本 = 3 本。header と合わせて 4 本で、境界の
# すぐ外側。以前は box で黙って切り詰められ (読み戻した要素が尾部を失う)、幅ゆえ
# に boxed method dispatch からも外れて、存在する `__repr__` が実行時 abort に
# なった。幅が分かるのは box の時点なので、そこで拒否する。
class U:
    def __init__(self, a: int | str) -> None:
        self.a: int | str = a

    def __repr__(self) -> str:
        return "U"


print([U(1)])
