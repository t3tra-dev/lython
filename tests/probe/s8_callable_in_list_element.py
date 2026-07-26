# probe: コールアブルを list 要素に格納して呼ぶ (フィールドなし)。**通る。**
#   これが S8 の論法の証拠側である -- **コンテナ要素スロットは既にコールアブルを
#   受け付ける**ので、フィールドが拒否されるのは表現の不在ではなく
#   フィールド経路の assignability の問題である。真の表現ギャップなら幅が判って
#   いる box の側で出るはずで、名前解決の側では出ない。
#   (出典: kernel-sidedefects)
# axes: width=callable op=store-into-container flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: hi

from typing import Callable


def make() -> str:
    return "hi"


fs: list[Callable[[], str]] = [make]
print(fs[0]())
