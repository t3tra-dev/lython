# payload box が持つのは 1 つのアドレスだけなので、1 lane より広い値は「その
# アドレスから残りを復元できる contract」でなければ入らない。クラスのインスタンス
# はフィールドを何本持っていても 1 lane (フィールドは body にあり、box には
# インスタンスのハンドルが入る) なので、ここに到達するのは union だけ。
#
# union フィールドは各メンバーが box に入るなら box 1 つで持つ
# (`classFieldStoredBoxed`)。`int | str` も `Node | Leaf` も `int | bool` も
# それで、リストにも入る。入らないのは自分の contract を持たないメンバー:
# `type[X]` の値は空 (どのクラスかは型が決めている) で、box に入れるものが無く、
# 読み戻すときに名指す class word も無い。だから `int | type[U]` のフィールドは
# tag + 各メンバーのレーンのまま inline に残り、そのインスタンスは header と
# 合わせて 3 lane に広がる。以前は box で黙って切り詰められ (読み戻した要素が
# 尾部を失う)、幅ゆえに boxed method dispatch からも外れて、存在する
# `__repr__` が実行時 abort になった。幅が分かるのは box の時点なので、そこで
# 拒否する。
class U:
    def __init__(self) -> None:
        self.a: "int | type[U]" = 1

    def __repr__(self) -> str:
        return "U"


print([U()])
