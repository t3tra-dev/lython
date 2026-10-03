# 非同期の組み直し: 中断できるコルーチンと asyncio

JavaScript の Promise を `await` で待つ (docs/js-host.md の段階 3) ために、Lython の
非同期の機構を、CPython と同じ形に組み直す。方針は利用者の判断による。JSPI で
今のモデルに中断だけを足すのではなく、コルーチンを本当に中断できるものにし、
asyncio を CPython の形で載せる。JSPI はその後で WASI 側に繋ぐ。

## 現状 (2026-10-03 時点の調査)

- **コルーチン**: 中断できない。`await coro` は、コルーチンの本体をその場で
  同期的に呼ぶだけ (`AwaitOps.cpp`)。`asyncio.sleep` は待たない。Future の結果は
  コンパイル時の evidence として扱われ、実行時には保持されない。
- **generator**: 本物の状態機械 (`GeneratorStateMachine.cpp`)。`send`、`throw`、
  `close`、try/finally をまたぐ yield、静的に分かる `yield from` に対応している。
  ただし次の制約がある。
  1. **再開は、作成元が静的に分かる場所に限られる。** `__step` を、作成時の
     SSA の引数値で静的に呼ぶ。引数として受け取った generator や、コンテナの
     中の generator は再開できない ("a generator returned out of a function
     cannot be resumed here")。
  2. **送る値と戻り値は int に限られる** (lane が i64)。
  3. **フレームに保持できる型の制約**: union、および lane の形を持たない型は、
     yield をまたいで保持できない。
  4. **yield の型**: すべての yield が同じ型 (lane の契約) でなければならない。
  5. **`yield from`**: 委譲先の本体をインライン展開する形でしか対応していない
     (generator オブジェクトへの実行時の委譲がない)。

## 目標の形 (CPython に倣う)

- コルーチンは generator と同じ状態機械に載せる。`await x` は
  `yield from x.__await__()` とする。
- Future、Task、イベントループは、`runtime/lib` の Python で書く。CPython の
  `asyncio` の純 Python 実装 (futures.py、tasks.py、base_events.py) を手本にする。
  - `Future.__await__` は、未完了なら自分自身を yield し、再開後に
    `self.result()` を返す。
  - `Task.__step` は、`coro.send(None)` / `coro.throw(exc)` でコルーチンを進める。
    yield された Future に完了時のコールバックを登録し、`StopIteration` で
    結果を受け取る。
  - ループは、ready キュー、`call_soon`、`call_later` (時刻順のヒープ)、
    `run_until_complete` を持つ。
- JS ホストでは、Promise を Future に橋渡しし、ループを JS のイベントループに
  接続する (WebLoop)。待つものしかないときは wasm から JS に戻る。

## 段階

1. **generator の動的な再開** (制約 1)。
   - 各 generator に、引数をフレームから読み直す `__step_stored` を作る
     (解放時の `__finalize` と同じ読み方)。
   - 値の lane の形が同じ generator の集まりごとに、フレームの target id で
     分岐するディスパッチャを作る。
   - 静的な作成元の evidence がない再開 (引数、コンテナ、フィールド) は、
     ディスパッチャを通す。
   - native でも有用で、ここで独立にコミットできる。
   - **済 (2026-10-03)**。`__step_stored` などの driver (`*_stored`) と、
     `__ly_generator_<step|advance|throw|close>$<yield の契約>` の dispatcher。
     作成元が分からない受け手への `next`、`for`、`send`、`throw`、`close`
     は、これを通る。候補は、yield の lane の契約が読む側と一致する
     generator。部分型を別の lane で yield するもの、状態機械でないものが
     候補に入り得るときは、名前を挙げて拒否する。
   - 残った穴:
     - protocol 型 (`list[Generator[...]]` と注釈したもの) の値は、
       runtime 契約が無いため所有権の対象外 (NonObject) になる。
       `pop` した値の参照が取られないので、再開は静的に拒否する。
       型を generator 関数から推論させれば (`tasks = [worker()]`) 動く。
       asyncio の Task はこの形に依存しないように書く。
     - 状態機械が `Iterator[...]` (protocol) の値を yield をまたいで保持
       できない (段階 2 のフレームの型)。
2. **送る値、戻り値、フレームの型を広げる** (制約 2〜4)。オブジェクトと None を
   送れるようにし、任意のオブジェクトを返せるようにする。union を含む値を
   フレームに保持し、yield の型を union にできるようにする。
3. **`yield from` の実行時の委譲** (制約 5)。インライン展開できない委譲先
   (generator オブジェクト、`__await__` の戻り値) を、実行時に駆動する。
4. **コルーチンを状態機械に載せる**。`async def` を generator として扱い、
   `await` を `yield from __await__()` にする。今の同期的な経路は廃止する。
5. **asyncio を Python で書く** (Future、Task、ループ、`sleep`、`gather`、
   `create_task`)。CPython と同じ出力の golden で確かめる。
6. **JS との接続**: Promise と Future の橋渡し、WebLoop。
7. **WASI 側で JSPI に繋ぐ**。

各段階で、native と全ターゲットの golden を通す。境界の外の形は、最も早い
静的な境界で拒否する。
