# payload box が持つのは 1 つのアドレスだけなので、1 lane より広い値は「その
# アドレスから残りを復元できる contract」でなければ入らない。クラスのインスタンス
# はフィールドを何本持っていても 1 lane (フィールドは body にあり、box には
# インスタンスのハンドルが入る) なので、ここに到達するのは union だけ。
#
# union フィールドは各メンバーが box に入り、box の class word でメンバーを
# 言い当てられるなら box 1 つで持つ (`classFieldStoredBoxed`)。`int | str` も
# `Node | Leaf` も `int | bool` も `int | type[A]` もそれで、リストにも入る。
# 言い当てられないのはクラス値のメンバーが 2 つあるとき: クラス値の box は
# どのクラスを指していても class word が `type` なので、`type[A]` と
# `type[B]` のどちらかを box からは決められない。だから
# `int | type[A] | type[B]` のフィールドは tag + 各メンバーのレーンのまま
# inline に残り、そのインスタンスは header と合わせて 5 lane に広がる。
# 以前は box で黙って切り詰められ (読み戻した要素が尾部を失う)、幅ゆえに
# boxed method dispatch からも外れて、存在する `__repr__` が実行時 abort に
# なった。幅が分かるのは box の時点なので、そこで拒否する。
class A:
    pass


class B:
    pass


class U:
    def __init__(self) -> None:
        self.a: "int | type[A] | type[B]" = 1

    def __repr__(self) -> str:
        return "U"


print([U()])
