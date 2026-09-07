# payload box が持つのは 1 つのアドレスだけなので、1 lane より広い値は「その
# アドレスから残りを復元できる contract」でなければ入らない。クラスのインスタンス
# はフィールドを何本持っていても 1 lane (フィールドは body にあり、box には
# インスタンスのハンドルが入る) なので、ここに到達するのは union だけ。
#
# union フィールドは各メンバーが box に入るなら box + class word で持つ
# (`classFieldStoredBoxed`)。`int | str` はそれで、リストにも入る。入らないのは
# メンバーの 1 つが entity を持たないとき: `builtins.bool` の runtime shape は
# i1 で、box の entity word に入れるアドレスがない。だから `int | bool` の
# フィールドは tag + 各メンバーのレーンのまま inline に残り、そのインスタンスは
# header と合わせて 4 lane に広がる。以前は box で黙って切り詰められ (読み戻した
# 要素が尾部を失う)、幅ゆえに boxed method dispatch からも外れて、存在する
# `__repr__` が実行時 abort になった。幅が分かるのは box の時点なので、そこで
# 拒否する。
class U:
    def __init__(self, a: "int | bool") -> None:
        self.a: "int | bool" = a

    def __repr__(self) -> str:
        return "U"


print([U(1)])
