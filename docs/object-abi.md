# オブジェクトとコンテナの ABI (v2)

コンテナの要素スロット、ヒープオブジェクトのレイアウト、確保器の設計を定める。
目標は三つで、優先順に **安全** (メモリ安全証明の前提を一つも崩さない)、**軽量**
(要素あたりのバイト数を CPython 以下にする)、**高速** (要素の出し入れで確保と
間接参照をしない)。

この文書は仕様であり、移行計画を含む。各フェーズの完了時に「現状」節の数字を
更新する。

## 1. 現状 (2026-10-06 計測、main eaf32113、AOT、Apple M 系、macOS)

### 1.1 要素あたりのメモリ (100 万要素、ピーク RSS から起動分を引いた値)

| プログラム | Lython | CPython 3.14 | 比 |
|---|---|---|---|
| `[0] * N` (スロットのみ) | 40 B | 8 B | 5.0 |
| `list[int]` (256 超の値) | 117 B | 40 B | 2.9 |
| `list[float]` | 102 B | 40 B | 2.5 |
| `list[str]` (6 文字前後) | 117 B | 56 B | 2.1 |
| `list[tuple[int, int]]` | 333 B | 101 B | 3.3 |
| 2 int フィールドのインスタンスの list | 271 B | 132 B | 2.1 |
| `dict[int, int]` | 233〜312 B | 110 B | 2.1〜2.8 |
| `dict[str, int]` | 310〜389 B | 135 B | 2.3〜2.9 |
| `set[int]` | 149〜264 B | 93 B | 1.6〜2.8 |

幅のある行は、macOS の大きなブロックのキャッシュ (§6.3) が効く場合と効かない
場合 (`MallocLargeCache=0`)。

### 1.2 速度

| プログラム | Lython | CPython |
|---|---|---|
| `list[int]` に 1000 万回 append して合計 | 0.72 s / 1151 MB | 0.32 s / 401 MB |
| `list[float]` に 500 万回 append して合計 | 0.45 s / 501 MB | 0.16 s / 209 MB |
| `dict[int, int]` を 200 万件作って合計 | 0.46 s / 466 MB | 0.10 s / 237 MB |
| インスタンス 200 万個の list を作って合計 | 0.39 s / 447 MB | 0.35 s / 248 MB |

append ループの時間の大半は合計側の `LyLong_Add`: 要素がヒープ int なので、
取り出した値が i64 レーンに乗らない。

### 1.3 原因 (どこにバイトがあるか)

**スロットが 5 ワード (40 B) で、うち 2 ワードは死んでいる** (`BoxLayout.h`)。

| ワード | 内容 | 実態 |
|---|---|---|
| 0 | refcount | スロットの中では誰も読まない。`LyObject_FromSlot` が単独 box に複写したとき 1 で上書きされるだけ (`CollectionPayload.cpp:487-494`) |
| 1 | class id | 本体。ハッシュ・比較・repr・解放の分岐に使う |
| 2 | 実体のアドレス | 本体。参照先ヘッダのオフセット 0 |
| 3 | 所有フラグ | 0 になるのは None、借用の一時 box、空にしたフィールドだけ。コンテナのスロットでは常に 1 |
| 4 | ハッシュのキャッシュ | dict と set の表が (state, hash) を別に持つ。list と tuple は読まない |

**要素の値は必ず別のヒープオブジェクト。** int (256 超) も float も、スロット
はそのポインタを持つだけで、値をスロットに置く経路が無い。

| オブジェクト | 中身 | 実際の確保量 |
|---|---|---|
| int (−5..256 以外) | ヘッダ 16 + meta 16 + 桁 | 80 B |
| float | ヘッダ 16 + f64 | 64 B |
| str (3 文字) | ヘッダ 16 + shape 8 + capacity 8 + 3 | 80 B |
| 2 int フィールドのインスタンス | ヘッダ 40 + 本体 (box 2 個 + 2 ワード) | 160 B + int 2 個 |
| tuple ヘッダ | 14 ワード (うち 6 ワードは幅を区別するための詰め物) | 128 B |

**確保器がオブジェクトごとに 32 B を足す** (`RuntimeSupportBuilder.cpp:553-948`)。
`LyMem_Alloc` はサイズクラスを 16 B の前置きに書き、`memref.alloc
{alignment = 16}` は位置合わせのため要求に 16 B を足す。CPython の pymalloc は
サイズクラスをプール (4 KiB) の見出しに持ち、ブロックに前置きを付けない。

### 1.4 フェーズごとの推移 (要素あたり、100 万要素、macOS、ピーク RSS)

| | 開始時 | P0 後 | P1α 後 | P1β 後 | P2 後 | P3 後 | P4 後 | P5 後 | 現在 | CPython |
|---|---|---|---|---|---|---|---|---|---|---|
| `list[int]` | 117 B | 85 B | 70 B | 24 B | 9 B | 9 B | 9 B | 9 B | 9 B | 40 B |
| `list[float]` | 102 B | 70 B | 55 B | 24 B | 9 B | 9 B | 9 B | 9 B | 9 B | 40 B |
| `list[str]` | 117 B | 85 B | 70 B | 70 B | 55 B | 55 B | 55 B | 55 B | 55 B | 56 B |
| `list[tuple[int, int]]` | 333 B | 269 B | 223 B | 223 B | 177 B | 177 B | 177 B | 116 B | 70 B | 101 B |
| 2 int フィールドのインスタンス | 271 B | 224 B | 177 B | 177 B | 132 B | 70 B | 70 B | 70 B | 70 B | 132 B |
| `dict[int, int]` | 312 B | 205 B | 193 B | 169 B | 137 B | 137 B | 97 B | 97 B | 77 B | 110 B |
| `set[int]` | 264 B | 127 B | 105 B | 72 B | 55 B | 55 B | 55 B | 55 B | 30 B | 93 B |

P1α はスロットを 5 ワードから 3 ワード (使わない refcount、class、実体) にした
段階: 所有フラグを廃し (実体がアドレスなら所有)、ハッシュを dict は並行配列に、
set は都度計算に移した。速度は append と合計 0.21 s、float 0.12 s、dict 0.11 s、
インスタンス 0.07 s (P0 と同じかそれより速い)。

P1β は int と float をスロットの実体ワードに即値で置く段階。ビット 0 が 1 なら
即値で、retain / release はもともとそのワードを飛ばす。int は 63 bit に収まる値
を `v << 1 | 1`、float は指数の上位 3 bit が 011 か 100 の値 (絶対値が
[2^-255, 2^256)) と +0.0 を Ruby の flonum と同じ回転で置く。それ以外 (-0.0、
NaN、inf、極端な指数、2^62 以上の int) はオブジェクトのまま。32 bit ターゲット
(wasm32、armv7) ではワードがアドレスを通ると上半分を失うので、int は 31 bit の
範囲だけ、float は即値にしない。

即値を書くのは「格納するだけ」の経路 (実行時モードの append / insert /
setitem / dict への格納 / set.add)。リテラルと evidence を持つ append はコンテナ
の中身の evidence も兼ねるので、従来どおりオブジェクトを置く (タプルとフィール
ドが変わらないのはこのため。P3 / P5 で扱う)。読む側はすべて `from_slot_word`
(即値ならオブジェクトを作り、アドレスなら retain する) を通る。manifest の
hash / eq は即値どうしを値で比べ、オブジェクトを作らない。

速度は append と合計 0.19 s、float 0.11 s、dict 0.11 s、インスタンス 0.07 s。

P2 はスロットを 1 ワードにした段階 (S4、§3 冒頭)。float の即値タグを `…10` に、
「アドレスでない」の判定を下位 2 bit に移し、class はワードから導く
(`__ly_slot_class`、lowering は `slotClassFromEntity`)。単独の `object` box は
自身がオブジェクトなので 5 ワード (refcount、class、実体) のままで、その word 2
へのポインタがスロットとして読める。速度は append と合計 0.18 s、float 0.11 s、
dict 0.11 s、インスタンス 0.06 s。

P3 はインスタンスのフィールドを対象にした段階。フィールドへの格納も「格納する
だけ」の経路に加え (int / float は即値で入り、その格納のフィールド evidence は
残さない: evidence が名指すオブジェクトをスロットが持たなくなるため)、本体を
ハンドルの word 3 から置いた (インスタンスが使うのは refcount、class、本体アド
レスの 3 ワードで、`builtins.object` ハンドル型の残り 2 ワードは誰も読まない 16 B
だった)。ハンドルの型を 3 ワードにするのは幅で解放関数を選ばなくなる P6 の後。
2 int フィールドのインスタンスは 1 個 70 B (CPython 132 B)、インスタンスのループ
は 0.05 s。

P4 は dict の表を詰めた段階。表の各スロットが (状態、ハッシュ) の 2 ワードを
持っていたのを状態 1 ワードにし、比較に使うハッシュは dict がすでに持つハッシュ
配列から読む (CPython のエントリがハッシュを持つのと同じ位置づけ)。表は容量
あたり 32 B から 16 B。`dict[int, int]` は 1 件 97 B (CPython 110 B)、速度は
変わらない。set の表は (状態、ハッシュ) のまま: set はエントリ側にハッシュ配列を
持たず、表から外すとハッシュの事前比較を失う (要素あたり 55 B で CPython の
93 B を下回っている)。

P5 は tuple を 1 回の確保にした段階。ハンドル (refcount、class、長さ、容量、要素
アドレス) と要素を同じブロックに置き、ハンドル型の 14 ワードのうち使わない
word 5..13 (解放関数を幅で選ぶための詰め物) は確保しない。要素は word 5 の位置
から始まる。`(i, i)` の list は 1 要素 116 B (CPython 101 B)。リテラルの要素は
evidence を兼ねるので即値にしない (残りの差の大半は要素の int オブジェクト)。

P6 は当初「幅による解放関数の区別 (`HandleWidthRegistry`) を class id に置き換え、
詰め物ワードを削除」だった。先に後半を行った: list (9 ワード中 5)、set
(11 中 9)、frozenset (13 中 9)、tuple (P5) のハンドルを、使うワードだけ確保する。
型は詰め物込みの幅のまま残し、詰め物は型の上にしか
無い (読み書きされないことを確保箇所以外の全アクセスで確認した)。list 1 個は
80 B から 48 B のブロックになり、`[[i] for i in range(10**6)]` は 117 MB
(CPython の tracemalloc で 104 MB)。

前半は後から行った。解放関数は contract 名だけで選ぶ (`findDeallocatorForValueGroup`
に名前の無い呼び出しは無い)。名前が無い・名前に合う解放関数が無い値は解放しない
側に倒れ、所有権検証器が拒否する。名前を揃えるために行ったこと:

- manifest の owned result 126 関数に `ly.ownership.owned_result_contracts` を宣言した。
- 例外クラスは別名表 (`ly.ownership.deallocator_aliases`) で
  `builtins.BaseException` の解放関数を引く。別名は名前でしか引かれない。
- 受け手の contract を結果の名前にするのは、initializer と
  `ly.runtime.result_evidence = "receiver"` だけにした。
- 引数の lane は宣言型の幅で数える。union 引数は tag と各メンバーの幅を合計する。
  以前は 1 lane と数えていたので、後続の引数の名前が 1 lane 以上ずれていた。
- protocol clone の引数には具体型 (`ly.ownership.protocol_argument_types`) を記録する。
- raise に渡す借用引数は、呼び出しの operand を形で走査せず、関数の引数グループ
  (宣言型で名前の付いたもの) と一致するかで判定する。
- 戻り値を走査して形の合う解放関数を探す処理 (`collectRuntimeResourceGroups`) は
  削除した。全 golden と examples で、名前で見つかる以上のグループを足していな
  かった。

幅を予約していた `HandleWidthRegistry.h` と、その検査 `abi.handle_width_reservations`
は削除した。

その後、ハンドル型の幅も使うワードまで縮めた。list 9→5、tuple 14→5、set 11→9、
frozenset 13→9、bytes 6→4、complex 7→4、`_js.JsProxy` 17→3。確保量が変わったの
は bytes (先頭 48→32 B) と complex (56→32 B) で、`complex` 100 万個は 71→41 MB
(CPython 38 MB)。他は以前から使うワードだけを確保していた。

幅が揃ったことで、型の一致を表現の一致とみなしていた判定が表に出た:

- container の判別 (`containerIsHandleFronted`) は「幅 8 以上」だったのを
  contract 名にした。
- `object` への upcast が、物理型の先頭一致で値をそのまま box として流していた。
  list の handle は box と同じ memref<5xi64> なので、グローバルに list の handle
  が入り、repr hook がその長さを entity として読んで落ちた。source class の
  インスタンスは以前から幅 5 で、`g: object = A(7); print(g)` と `object` を返す
  関数の戻り値は main でも SIGSEGV だった。upcast 先が `object` で元が別の class
  なら、型が一致しても alias にしない (box は消費側が作る)。

即値を読むたびにオブジェクトを作るので、`d[k]` の読み出しが多いループは P1α と
同程度にとどまる (読み出しで evidence に直接載せるのは P2 の f64 / i64 evidence
と合わせて扱う)。

速度 (最良 5 回): append と合計 0.21 → 0.23 s、float 0.13 → 0.13 s、dict
0.11 → 0.11 s、インスタンス 0.07 → 0.08 s。確保器の遅い経路
(`LyMem_LargeAlloc` / `LyMem_MapAlloc` / `LyMem_Refill`) を out-of-line に
保ち、`aligned_alloc(16, n)` を `LyMem_Alloc(n)` に直接書き換えて、ここまで
戻した。

## 2. 不変条件 (新 ABI が守るもの)

S1〜S6 はどのフェーズでも成り立たなければならない。破るフェーズは差し戻す。

**S1. アドレスはオフセット 0。** スロット・フィールド・ヘッダに入るアドレスは、
生きている確保領域の先頭を指す。証明の `site-address-recovers`
(`proof/src/Proof/RC/Address.agda:111-127`) はこの場合だけを扱う。要素配列の
途中を指すアドレスを、保持されるワードとして書かない。

**S2. 所有は型とビットで決まる。** スロットが参照を 1 つ持つかどうかは、そのス
ロットの静的な格納種別 (§3) と、タグ付き種別ではタグの値だけで決まる。実行時
の所有フラグは持たない。参照を持つスロット 1 つにつき、参照先の refcount に
ちょうど 1 が積まれている (証明の `field′` の多重度、`Aggregate.agda:15-20`)。

**S3. 伸びるバッファから派生した見方は、伸ばしうる操作の後に作り直す。**
list の items 配列、dict の entries と index 表、set の表は、ヘッダが名指す別の
確保領域であり、伸長で場所が変わる (証明の `resizeBuffer`、
`Object/Ops.agda:164-172`)。そこから作った memref や要素アドレスを、伸長をまた
いで使わない。ヘッダ自体は動かない (`realloc-of-buffer-leaves-the-box`)。

**S4. スロットのワードは自分で種別を言う。** (P2 で改訂。当初は「格納種別は
コンテナの静的型の関数で、型の消えた経路のためにヘッダが要素種別を 1 バイトで
記録する」だった。) スロットは 8 B の 1 ワードで、下位 2 bit がそのワードの
種別を決める: `00` かつ 0 なら None、`00` で 0 以外ならオブジェクトのアドレス
(class id はそのヘッダの word 1)、`…1` は即値の int、`…10` は即値の float。
型の消えた経路も静的型の分かる経路も同じ読み方をするので、種別バイトも静的種別
ごとの関数の版も要らない。

**S5. 値型の同一性は観測できない。** int・float・str・bytes・complex に対する
`is` はすでに emit 時に拒否されている (`EmitterExpressions.cpp`、"identity of
value types is an implementation detail")。`id()` は無い。したがって値をスロッ
トに直接置いても、取り出すたびに別のオブジェクトになっても、プログラムからは
見えない。この拒否を緩めるときは、この節を先に改める。

**S6. 不死オブジェクト。** refcount が INT64_MAX のオブジェクト (小さい int の
表、bool、死んだ枝の置き物) は retain / release で触らない。新しく作る静的
オブジェクトも同じ印を使う。

## 3. 値の格納種別

**実装された形 (P2)。** すべてのスロットは 8 B の 1 ワード (S4)。下の表と §3.1
は P2 以前の提案で、静的型ごとに幅を変える代わりに、ワードの下位 2 bit で種別を
自己記述する形に置き換えた。理由: (1) class ワードは常に「実体ヘッダの word 1」
か「即値のタグ」の写しで、持つ必要がなかった、(2) 型の消えた経路のために種別
バイトと汎用版の関数を別に持つより、全経路が同じワードを同じ規則で読むほうが
食い違いの余地が無い、(3) `str` や インスタンスの要素も 8 B になる (静的種別案
では `Ref` で同じ 8 B だが、union や `T | None` は 16 B だった)。`Bool` の 1 B は
採らない (bool は不死の単一オブジェクトのアドレスで 8 B)。

スロット (コンテナの要素、インスタンスのフィールド、クロージャのセル、generator
の frame の退避先) に値を置く形式を、静的型ごとに 1 つ決める (以下は P2 以前の
提案)。レジスタ上の
レーン (lowering の `RuntimeBundle`) はこの節の対象外で、格納と取り出しの境界
で変換する。

| 種別 | 静的型 | 幅 | 中身 | 参照を持つか |
|---|---|---|---|---|
| `Int` | `int` | 8 B | 下位ビット 1: 63 bit 整数を即値で (`v << 1 \| 1`)。下位ビット 0: 大きい int (`LyLong`) のアドレス | ビット 0 が 0 のとき |
| `Float` | `float` | 8 B | f64 のビット列 | 持たない |
| `Bool` | `bool` | 1 B | 0 / 1 | 持たない |
| `Ref` | 具体的な参照型 1 つ (`str`、`list[...]`、インスタンス、tuple ...) | 8 B | 実体ヘッダのアドレス (0 にならない) | 常に |
| `OptRef` | `T \| None` (T が `Ref` 種別) | 8 B | アドレス、または 0 = None | 0 でないとき |
| `Value` | union、`object`、protocol、`Callable`、`Int`/`Float` を含む Optional | 16 B | タグ 8 B + 中身 8 B (§3.1) | タグが参照を示すとき |

**`Int` の即値幅が 63 bit である理由。** 8 B に収めると判別に 1 ビット要る。
`Ly_IncRef` は既に「アドレスのビット 0 が立っていたら何もしない」分岐を持つ
(`builtins.mlir:1497-1504`) ので、retain / release はそのまま使える。63 bit を
超える値だけが `LyLong` を確保する (CPython が int オブジェクトを常に確保する
のに対し、ほとんどのプログラムで 0 回)。

### 3.1 `Value` (16 B)

| タグ | 中身 | 参照 |
|---|---|---|
| `None` | 0 | なし |
| `Bool` | 0 / 1 | なし |
| `InlineInt` | i64 そのもの (64 bit 全域) | なし |
| `InlineFloat` | f64 のビット列 | なし |
| class id (それ以外すべて) | 実体ヘッダのアドレス | あり |

タグの値は class id と同じ空間に置き、即値用の 4 つは class id として使われて
いない番号を予約する。`Value` は今の 5 ワード box の役目 (型の消えた値の運搬)
を引き継ぐ。単独の box をヒープに確保する経路 (`RuntimeABI.cpp:1030-1045`、
64 B) は無くなり、`object` 型の値はレジスタ上の 2 ワードになる。

### 3.2 型と種別の対応で迷う場合

- `list[int | None]` は `Value` (即値と None を同じ 16 B で表せる)。
- `list[Animal]` で `Animal` にサブクラスがあっても `Ref`。どのクラスかは実体
  ヘッダの class id が言う。
- `list[str | bytes]` は `Value`。
- 要素型が型変数のままのコンテナは作られない (ジェネリックは特殊化済み)。

## 4. オブジェクトのレイアウト

### 4.1 共通ヘッダ (16 B)

| オフセット | 内容 |
|---|---|
| 0 | refcount (i64、原子的に増減。INT64_MAX は不死) |
| 8 | class id (下位 32 bit) と、コンテナの要素種別などの小さい属性 (上位 32 bit) |

解放関数は class id で選ぶ。今のように「ヘッダの幅」で解放関数を区別するため
の詰め物ワード (list の word 8、tuple の 8〜13、set の 9〜10 など、
`HandleWidthRegistry.h`) は持たない。

### 4.2 各型

| 型 | v2 のレイアウト | 実際の確保量 |
|---|---|---|
| int (63 bit 超のみ) | ヘッダ + 符号と桁数 8 + 30 bit 桁 | 桁数次第 |
| float (`Value` から追い出すときのみ) | ヘッダ + f64 | 32 B |
| str | 今と同じ (ヘッダ + shape 8 + capacity 8 + 符号単位)。capacity は `s += x` をその場で伸ばすために残す | 8 文字以下で 48 B (今は 80 B) |
| bytes | ヘッダ + 長さ 8 + バイト列 (今の 6 ワード handle と別バッファを 1 つの確保に) | 長さ次第 |
| list | ヘッダ + 長さ + 容量 + items アドレス = 40 B。items は要素種別の幅 × 容量 | 48 B + 配列 |
| tuple | ヘッダ + 長さ 8 + 要素をヘッダの直後に並べる (不変なので 1 回の確保)。位置ごとの静的型で種別を決める (`tuple[int, str]` は `Int` + `Ref`) | `(int, int)` で 48 B |
| dict | CPython 3.6 以降と同じ compact 形式: ヘッダ + 長さ + 使用数 + index 表と entries (1 つの確保)。index は表の大きさに応じて 1/2/4/8 B。entry は (hash 8 B、キー、値) をそれぞれの種別の幅で | `dict[int, int]` で 1 件 約 40 B |
| set | ヘッダ + 長さ + fill + mask + 表アドレス。表は (hash 8 B、要素) | `set[int]` で 1 件 約 27〜53 B (負荷率 60% の開番地法) |
| インスタンス | ヘッダの直後にフィールドを宣言順に並べる。フィールドごとに §3 の種別 (今の「本体アドレス」ワードと、bool 以外すべてを box にする規則は無くなる) | 2 int フィールドで 32 B |
| generator | 状態ワードと、frame の退避先を種別ごとに | 退避する値次第 |
| 関数 / クロージャ | ヘッダ + target id + defaults + セル数 + セル (種別ごと) | セル数次第 |

### 4.3 見込み (要素あたり、確保器の変更 §6 込み)

| | 現状 | v2 | CPython |
|---|---|---|---|
| `list[int]` | 117 B | **8 B** | 40 B |
| `list[float]` | 102 B | **8 B** | 40 B |
| `list[str]` (6 文字前後) | 117 B | **56 B** | 56 B |
| `list[tuple[int, int]]` | 333 B | **56 B** | 101 B |
| 2 int フィールドのインスタンスの list | 271 B | **40 B** | 132 B |
| `dict[int, int]` | 233〜312 B | **約 40 B** | 110 B |
| `set[int]` | 149〜264 B | **約 27〜53 B** | 93 B |

速度面では、`Int` / `Float` 種別の格納と取り出しが確保をしない。取り出した値は
レジスタ上の i64 evidence (`primitiveI64`) にそのまま乗るので、上の合計ループ
は `LyLong_Add` を呼ばない。f64 にも同じ evidence が要る (§7 P2)。

## 5. 安全性の根拠

- **S1** は種別の定義から出る。アドレスを持つのは `Int` (ビット 0 が 0)、`Ref`、
  `OptRef`、`Value` (参照タグ) だけで、いずれも `LyMem` が返した確保の先頭、
  つまり実体ヘッダを指す。tuple とインスタンスは要素・フィールドを自分の確保
  の中に持つが、それを指すアドレスはどこにも保持しない (オフセットはコンパイル
  時定数)。
- **S2** により、今の「所有フラグが 0 のスロット」は消える。None は `OptRef`
  の 0 か `Value` の `None` タグ、借用の一時 box (検索キー) は格納でなく
  レジスタ上の値として渡す。証明の `field′` は「参照を持つスロット」の数え方
  をそのまま使う。
- **S3** は今の慣習 (`CollectionPayload.cpp:950-954`) を規則に格上げする。
  verifier の内部ビュー検出 (`collectBoxWordDerivedViews`、
  `common/Ownership.cpp:692-755`) は、ヘッダから読んだバッファアドレスから
  作った memref を「実体の内部」として固定する。新しい種別でもこの導出の形
  (ヘッダの load → inttoptr → memref) を変えない。歩幅が種別ごとに変わるだけ。
- **S4** の要素種別バイトは、型の消えた経路が中身を読み違えないためにある。
  静的型が分かっている経路は読まず、コンパイル時の種別で直接読む。
- 証明側 (`proof/src/Proof/Object/Layout.agda`) は「固定ヘッダ + 別バッファ」を
  すでにモデル化しており、v2 の list / dict / set はそれに近づく。即値スロット
  は「参照を持たないワード」で、`Aggregate` の数え方の外にある。証明の更新は
  §7 の各フェーズと同時に行う。

## 6. 確保器

### 6.1 前置きの廃止

サイズクラスを各ブロックの前置き 16 B でなく、プールの見出しに置く。プールは
16 KiB に揃えて確保し、解放時はアドレスの下位ビットを落としてプールを得る
(pymalloc と同じ)。16 B 単位、512 B 以下が対象。

### 6.2 位置合わせの水増しの廃止

プールのブロックは最初から 16 B 境界にある。`memref.alloc {alignment = 16}` を
`LyMem_Alloc` に直接下げ、要求に 16 B を足す一般の下げ方を通さない。

### 6.3 大きなブロックを抱えない

- macOS では、1 MiB 以上のブロック (list の items、dict、set の表など) を mmap で
  確保し、伸長は新しい領域へのページ単位の `vm_copy`、解放は munmap にする。
  libmalloc は realloc 途中の大きなブロックを解放後も抱えるため。C で再現した
  測定: 20 万要素を 100 回作り直すループで 503 MB → 11 MB、2000 万要素 1 本で
  2294 MB → 766 MB (最終サイズ 800 MB)。Linux の glibc は同じ測定で 8 MB /
  763 MB なので、この経路は macOS だけ。`malloc_zone_pressure_relief` は効果が
  無かった。
- 512 B を超えるブロックだけが前置き 16 B (種別と容量) を持つ。プールの
  ブロックかどうかはアリーナの地図で先に判定するので、前置きを読むのは前置き
  のあるブロックだけ。

### 6.4 空いたアリーナは返さない (P0 の時点)

プールを返すにはプールごとの使用数を確保の経路で数える必要があり、以前の
計測ではそれだけでコンテナのベンチが 10〜30% 遅く、返せたのは 0.5 MB だった
(`RuntimeSupportBuilder.cpp` の確保器のコメント)。当時は要素 box が 512 B の
上限を超えてプールに入らなかったのが返せない理由で、P1 / P2 でスロットが
縮むと事情が変わる。プールの見出しはそのための土台として既にある。P2 の後で
測り直して決める。

### 6.5 refcount の原子性は変えない

スレッド安全性の verifier (`verifier/runtime/ThreadSafe.cpp`) は refcount の
増減を `generic_atomic_rmw` の形 (本体の中の正値検査を含めて) で照合している。
測定 (M 系、競合なし) では、比較交換のループ 0.84 ns、`ldadd` 0.68 ns、原子で
ない加算 0.67 ns で、確保と間接参照に比べて小さい。verifier の照合を書き換え
てまで `ldadd` にする利得は無いので、今の形のままにする。

## 7. 移行計画

各フェーズの完了条件は共通で、両ビルドの全件スイート、leak ラベル、AOT と wasm
の golden、`LYTHON_PERF` での主要ケースの所要時間、§1 の表の再計測。

| フェーズ | 内容 | 主な触る場所 | 期待効果 |
|---|---|---|---|
| P0 | 確保器 (§6.1〜6.3) | `RuntimeSupportBuilder.cpp`、`LLVMFinalize.cpp`、`memref.alloc` の下げ方 | 全オブジェクト −32 B、macOS の大きな list |
| P1 | スロットを `Value` (16 B) に統一。所有フラグ・死んだ refcount ワード・ハッシュワードを廃止し、即値の int / float / bool / None をスロットに置く | `BoxLayout.h` と box ワードの直書き箇所 (§8)、`__ly_box_*`、dict / set の表 | スロット 40 → 16 B、int と float の要素の確保が 0 |
| P2 | (実施: すべてのスロットを自己記述する 8 B の 1 ワードに。当初案の静的種別・種別バイト・`Bool` 1 B は不要になった) | `BoxLayout.h`、manifest の box 読み書き、class ワードの読み手 | スロット 24 → 8 B |
| P3 | (実施: フィールドを即値で格納、本体をハンドルの word 3 から) | `AttributeOps.cpp`、`Manifest/Calls.cpp`、`RuntimeABI.cpp` | 2 int フィールドのインスタンス 132 → 70 B |
| P4 | (実施: dict の表を状態 1 ワードに。set は据え置き) | dict の manifest 実装 | `dict[int, int]` 137 → 97 B |
| P5 | (実施: tuple を 1 回の確保に) | tuple の manifest 実装 | `list[tuple[int, int]]` 177 → 116 B |
| P6 | (実施: 詰め物ワードを確保しない。解放関数の選択は幅のまま) | list / set / frozenset の確保 | list 1 個 80 → 48 B |

P1 を P2 より先にするのは、P1 が型を問わず一様に効き、P2 の特殊化を後から
足しても `Value` が汎用経路として残るため。P2 から始めると、型が消えた経路の
ために P1 相当の汎用形式をどのみち先に用意することになる。

## 8. 書き換えの範囲 (2026-10-06 時点)

- manifest: `__ly_box_word_count` の呼び出し 84 か所、`memref<5xi64>` の署名
  173 か所 (`builtins.mlir`)、29 か所 (`types.mlir`)、ワード 1 / 2 の直書き
  約 20 か所。
- `RuntimeSupportBuilder.cpp` のスロット retain / release
  (`:1510-1725`)、`TracebackSupportBuilder.cpp:2256,2261`。
- lowering: `CollectionPayload.cpp`、`ContainerInterior.cpp`、`RuntimeABI.cpp`、
  `AttributeOps.cpp` (約 20 か所)、`GetItemOps.cpp`、`SpecialMethodOps.cpp`、
  `CallableOps.cpp`、`IndirectCallableOps.cpp`、`PackAndBindingOps.cpp`、
  `GeneratorStateMachine.cpp`。
- 固定値の検査: `DriverTests.cpp:1149-1163` (box の幅とワード位置)、
  `:1186-1293` (歩幅の直書き禁止)。
- 古い記述 (今のうちに直す): `BoxLayout.h:3-8` (12 ワード)、`BoxLayout.cpp:44`
  と `Verification.cpp:477` (16 ワード)、`HandleWidthRegistry.h:68,332`
  (`builtins.object` 16)、`Lowerer.h:164-166` (int をヘッダワードに置く記述)、
  `docs/bigint-design.md` の 3 レーン記述、`proof/src/Proof/Memory/Lython.agda:21-24`。
