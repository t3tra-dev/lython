# 任意精度 int: tagged 表現と昇格の設計

rfc/stdlib-semantics.md R1 (任意精度昇格) の実装設計メモ。現状調査 (Wave 0 M1)
の結果と、tagged 表現をどこに置くかの決定を記録する。

## 現状 (調査結果)

int の「tagged 表現」は既に二層で存在する。新しいボックス形式を発明する必要は
なく、欠けている演算を既存表現の上に埋める。

### 層 1: コンパイル時 evidence (SSA レーン)

`RuntimeBundle::primitiveI64` (`lowering/Passes/Runtime/Model/Bundles.h`) が
int 値ごとに `{value: i64, valid: i1}` を運ぶ。これが fast path のタグに相当
する:

- `py.int.constant` は i64 に収まる限り `valid = true` の evidence だけを作り、
  オブジェクト化は遅延 (`hasLazyPrimitiveI64Object`)。
- `__add__`/`__sub__`/`__mul__` と比較は `I64Calls.cpp` が
  「evidence が有効 かつ 演算が溢れない」分岐 (`scf.if`) を張り、
  then 側は素の `arith` 演算、else 側は manifest メソッド呼び出し
  (`LyLong_*`) に落ちる。結果 bundle は物理値 (scf.if の結果) と
  `{rawResult, fastValid}` evidence の両方を持つ。
- overflow 検出は add/sub が符号比較、mul が `arith.mulsi_extended`。
  (llvm.*.with.overflow は LLVM 変換後に同等の命令選択がされるため
  arith レベルの表現で十分。)

### 層 2: ランタイムオブジェクト (heap 表現)

`builtins.int` の物理型は `(header memref<2xi64>, meta memref<2xi64>,
digits memref<?xi32>)`。CPython の longobject と同じ 30-bit digit
(基数 2^30、リトルエンディアン limb 列):

- `header` = [refcount, layout]。0/1/2 は immortal キャッシュ
  (refcount = INT64_MAX の memref.global)。
- `meta` = [sign (-1/0/+1), digit_count]。
- 単一アロケーション: [0,16) header / [16,32) meta / [32,..) digits。
- 各エントリポイント (`LyLong_Add` 等) 内にも「両オペランドが i64 に収まり
  結果も収まるなら素の i64 演算」という第二の fast path がある。

つまり **tagged = (i64 evidence lane) + (digits オブジェクト)** で、昇格は
「evidence の valid が落ちた瞬間に manifest 呼び出しへ折り返す」ことで起こる。
この設計を維持し、以下のギャップだけを埋める。

## ギャップ (M1 調査で確認した誤動作・未実装)

| 項目 | 現状 | 方針 |
| --- | --- | --- |
| `**` (Pow) | emitter が未知の BinOp を `__add__` に **フォールスルー** し `2**3 == 5` | Py_PowOp を追加し `__pow__` へ。未知 op は診断 (never silently mis-execute) |
| 大整数リテラル | lowering が明示拒否 | コンパイル時に 30-bit digits へ分解し memref.global + `from_digits` primitive で構築 |
| `//` `%` (>i64) | `LyLong_AsI64` で **無言切り捨て** (30! // 10^6 が負値) | 一般 divmod (digit 長除算) を実装 |
| `INT64_MIN // -1` | i64 divsi の UB (INT64_MIN を返す) | 小径路のガードに除外条件を追加 |
| `<<` `>>` | i64 のみ。`1 << 100` が **無言 wrap** (2^36) | digit shift + 端数 bit shift の一般実装 |
| `& \| ^` (>i64) | `AsI64` 切り捨て | 2 の補数 digit 演算 (CPython longobject 方式) |
| `/` (>i64) | `AsI64` 経由で誤値 | `AsF64` を digit 累積に一般化 |
| `__hash__` | manifest 宣言のみで実装なし。`hash()` builtin も未配線 | CPython 準拠 (2^61-1 modular reduction、-1 → -2)。float 側が将来 `hash(1.0)` を同じ値に合わせられる形 |
| `int(str)` | 未実装 (診断) | 10 進 (符号 + underscore) の任意長パース |
| `round(big, -n)` | `AsI64` 切り捨て | fits ガード + 一般 divmod 経由 |

floor 除算の負数丸めは **i64 範囲内では既に CPython 準拠** (`-7 // 2 == -4`,
`-7 % 2 == 1`; sdiv/srem に符号補正を挿入済み) — golden テストで固定する。

## 一般 divmod のアルゴリズム選択

除数 1 digit: MSB→LSB の 64-bit 逐次除算。多 digit: **ビット単位の
shift-subtract 長除算** (O(bits·m))。Knuth Algorithm D (O(n·m)) を選ばなかった
のは、商推定の補正ループを手書き MLIR (scf/arith) で正しく書き切る検証コスト
が高く、まず正しさを固定したいため。定数倍 (~30x) は digit 数十個規模の
golden テストでは問題にならない。プロファイルで律速になった時点で D に差し
替える (エントリポイントの契約は変わらない)。

## 意味論ノート (CPython 3.14 準拠の適用)

- `int ** 負の int` は CPython では float を返すが、静的型 (`int.__pow__:
  (int, int) -> int`) と両立しないため実行時 ValueError で拒否し、
  `float(base) ** exp` への書き換えを促す (docstring 逸脱ルール適用)。
- `hash(x)`: `x % (2^61 - 1)` を digit 列上で還元、負数は符号適用後
  `-1 → -2`。SipHash ランダム化は str/bytes の話で int には適用されない。
- 大整数 → float (`/` や `float()`) は digit 累積のため correctly-rounded
  でない場合が 1 ulp 程度あり得る (CPython は正確丸め)。逸脱として記録。
