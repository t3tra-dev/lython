# payload box が持つのは 1 つのアドレスだけなので、1 lane より広い値は「その
# アドレスから残りを復元できる contract」でなければ入らない。クラスのインスタンス
# はフィールドを何本持っていても 1 lane (フィールドは body にあり、box には
# インスタンスのハンドルが入る) なので、ここに到達するのは union だけ。
#
# union の storage は tag + 各メンバーのレーンで、メンバー同士は entity を共有
# しないから 1 つのアドレスがそれらを名指せない。int | str = tag + int 1 本 +
# str 1 本、header と合わせて 5 本。以前は box で黙って切り詰められ (読み戻した
# 要素が尾部を失う)、幅ゆえに boxed method dispatch からも外れて、存在する
# `__repr__` が実行時 abort になった。幅が分かるのは box の時点なので、そこで
# 拒否する。
class U:
    def __init__(self, a: int | str) -> None:
        self.a: int | str = a

    def __repr__(self) -> str:
        return "U"


print([U(1)])
