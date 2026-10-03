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
   - **済 (2026-10-03)**: 戻り値は lane の形を持つ任意の契約になった
     (`returnLane`)。`__step` は戻り値を解放し、`__step_full` は所有権付きで
     返す。union は payload box に詰めて `builtins.object` の lane で中断を
     またぐ。対象は、引数、frame、yield する値、戻り値。union の読み取りに
     member 型の generator が来る場合は、box に詰める adapter を挟む。
   - **未 (後回し)**: int 以外のオブジェクトを `send` すること。asyncio の
     Task は `send(None)` と `throw` しか使わないので、急がない。
3. **`yield from` の実行時の委譲** (制約 5)。インライン展開できない委譲先
   (generator オブジェクト、`__await__` の戻り値) を、実行時に駆動する。
   - **済 (2026-10-03)**: 展開できない `yield from` は、`py.generator.step`
     (frame の dispatcher で 1 回進める) と、委譲の印付きの `py.yield_value`
     のループになる。外側に throw/close が届くと、inject として委譲先に
     転送される。静的な展開は 16 回までで、自分自身には行わない (再帰する
     委譲は、実行時の委譲になる)。
4. **コルーチンを状態機械に載せる**。`async def` を generator として扱い、
   `await` を `yield from __await__()` にする。今の同期的な経路は廃止する。
   - **済 (2026-10-04)**。`async def` は、yield が object、send が None、
     戻り値が R の generator 本体になる (公開の型は
     `types.CoroutineType[object, None, R]`。Phase 8d が
     `types.GeneratorType` に書き換える)。yield を持たないコルーチンも状態機械に
     載せる。`await x` は、x がコルーチンならそれ自身への、そうでなければ
     `x.__await__()` への `yield from` になる。`async with` は
     `__aenter__`/`__aexit__` を、`async for` は `__aiter__`/`__anext__` を、
     それぞれ Python で書いたクラスのメソッドとして呼んで await する。
     中断し得る dunder は、本体をインライン展開せずに symbol で呼ぶ
     (インライン展開すると、呼んだ場所で本体を最後まで実行してしまう)。
   - 旧モデル (`py.await`、`py.aenter`、`py.aexit`、`py.aiter`、`py.anext`、
     AsyncThunk、`modules/asyncio.mlir`、`_asyncio.mlir`、coroutine の
     evidence) は削除した。manifest のクラスで `async with` / `async for` を
     使うと、欠けている dunder を名指しで拒否する。
   - 拒否するもの: async generator (`async def` の中の `yield`。定義の場所で
     拒否する)、async の `__aiter__`、async for/else、async for の中の return。
5. **asyncio を Python で書く** (Future、Task、ループ、`sleep`、`gather`、
   `create_task`)。CPython と同じ出力の golden で確かめる。
   - **済 (2026-10-04)**: `src/lython/runtime/lib/asyncio.py`。CPython の
     events.py、base_events.py、futures.py、tasks.py の移植。CPython からの
     逸脱は、ファイル冒頭の docstring に列挙してある。主なもの:
     - Task は Future の派生ではない。両方とも `_Waiter` から派生する。
     - コールバックは引数を取らない。
     - Task は、コルーチンの結果を `StopIteration.value` からではなく、
       それを await する包みのコルーチンから受け取る。
     - ループは `create_future()` を持たない。
   - golden: `asyncio_tasks_interleave_and_settle` (交互の実行、gather、
     タイマーの順序、例外、cancel、Future)、
     `an_async_for_suspends_inside_anext`。
   - 途中で直した欠陥:
     - 空の union member から例外の message を読んでいた
       (`BaseException | None` が None のときの segfault)。
     - 関数参照が await をまたいで生きていた。
     - 同じ値を 2 本の block 引数に渡す edge で、所有権の挿入と検証が
       食い違っていた。挿入の前に 1 本にまとめる。
     - union 読み出し用の box adapter が、StepFull の戻り値の所有権を
       落としていた。
   - 残った穴 (asyncio はこれらに依存しないように書いてある):
     - generic クラスが、自分の型引数を持つ generic な基底から派生できない
       (`class B(A[T])`)。
     - 基底クラスが宣言される前に特殊化された generic クラスは、基底の
       フィールドを持たない。
     - generic なサブクラスに対する仮想ディスパッチがない。
     - `isinstance(o, int)` で object を絞り込めない。
     - int 以外のオブジェクトを `send` できない (段階 2 から持ち越し)。
6. **JS との接続**: Promise と Future の橋渡し、WebLoop。
   - **済 (2026-10-04)**。asyncio.py の `if sys.platform == "emscripten":`
     分岐に置いた (別モジュールにすると asyncio と import が循環する)。
     - ループは Pyodide の WebLoop と同じく、ホストのループでもある。仕事が
       あって誰もループを回していないとき、`setTimeout` でホストに呼び戻しを
       頼む (`_wake_host` / `_host_tick`)。ホスト上のループは常に実行中と
       みなすので、module レベルの `create_task` が使え、その task は main の
       本体が返った後にホストのループで進む。
     - `await promise` は、emitter が `asyncio._host_future(promise)` の
       呼び出しに書き換え、返った Future を待つ。解決値は Promise の型引数で
       検査し (`then` のコールバックの引数の変換)、食い違いも reject も
       `catch` で Future に届ける。reject は
       RuntimeError("<名前>: <メッセージ>")。
     - `asyncio.run()` はホスト上でもブロックする。Python だけを待つ間は
       動き、ホストしか進められない状態 (ready もタイマーもない) になると
       RuntimeError にする (JSPI の段階で、ここが中断になる)。ホストの呼び
       戻しの中から呼ぶと、CPython と同じく「実行中のループ」で拒否する。
   - そのために直したこと:
     - ドライバの import 収集が、`sys.platform` の比較で決まる module
       レベルの分岐を、ターゲットで折り畳む。死んだ分岐の `import js` で、
       native が拒否されなくなった。
     - generic な stub クラス (`class Promise[T]`) の型パラメータを登録し、
       メンバーを読む間は型変数として束縛する。`Promise[int]` のメソッドが
       解決できなかった。注釈の `Promise[int]` (グローバルと同名のクラスの
       エイリアスへの添字) も読めるようにした。
     - stub が `number` (float) と宣言した引数は int も受ける
       (`setTimeout(f, 10)`)。値は自分の型で境界を越える。
     - 式文で捨てられるホスト呼び出しの結果は、検査も変換もしない (node の
       `setTimeout` は number ではなく Timeout を返す)。
     - 局所変数が、同名で import された module を隠す。module 名の集合に
       スコープがないため、`import time as t` で time.py の `mktime(t)` が
       壊れていた。
   - 残った穴:
     - 戻り値にしか現れないメソッドの型変数 (`Promise.reject[T](...)`) は、
       注釈の文脈から決まらず拒否される。
     - Promise 以外の thenable への await は拒否する。
7. **WASI 側で JSPI に繋ぐ**。
   - **済 (2026-10-04)**。
     - `lyc --target wasm32-wasip1 --js-host -o prog.js` で、WASI のプログラムを
       JS ホストの下で動かす。出力は `prog.wasm` と、Emscripten を使わない自前の
       ローダ `prog.js` (CommonJS。`lython_js.js` と
       `runtime/js/lython_wasi.js` から成る)。ローダは WASI preview-1 の呼び出し
       (stdio、時計、乱数、引数、`poll_oneoff`、`proc_exit`。ファイルは開けない)
       に答え、`lython_js` の import に JS 側の本体を渡す。
     - LLVM の段で、`LyJs_*` の宣言に `wasm-import-module = "lython_js"`、
       JS から呼ばれる入口に `wasm-export-name` を付ける
       (`installJsHostEntryPoints`)。
     - `sys._js_host`: JS ホストがあるか (Emscripten、または `--js-host` 付きの
       WASI)。`sys.platform` では WASI のホストの有無を言えないので、Lython 独自の
       静的な定数にした。ドライバの import 収集と emitter の分岐の両方で
       折り畳む。
     - JSPI: ローダは `poll_oneoff` (`time.sleep`) と `LyJs_WaitForHost`
       (`_js.wait_for_host`) を `WebAssembly.Suspending` で包み、`_start` を
       `WebAssembly.promising` で呼ぶ。眠る間も、ループがホストを待つ間も、JS の
       イベントループが進む。ホストが呼び戻すと (コールバックの終わりで) 待ちが
       解ける。
       - コールバックの最中は、ホスト自身のスタックなので中断しない
         (`wait_for_host` は False を返し、`sleep` は空回りする)。
     - asyncio: ループがホストしか進められない状態になると、`_host_wait` で
       中断して待つ。中断できない所 (Emscripten、JSPI のない環境) では、
       これまでどおり RuntimeError にする。タイマーを待つ間も、ホストを待つ
       ので、待っている間に Promise が解決できる。
       - `asyncio.run()` が Promise を await できるようになった
         (golden `a_blocked_program_lets_the_host_run_under_jspi`)。
   - 途中で直したこと: Future が完了時のコールバックを、その時点の
     `get_event_loop()` に積んでいた。`run()` が失敗した後に Promise が
     解決すると、放棄された task が次のループで再開していた。CPython と
     同じく、作られたループに結び付けた。
   - テスト: `tests/golden/js` の各ケースを、Emscripten (`js`) と WASI の
     ローダ (`jspi`) の両方で走らせる。答えが違うケースは `.jspi.stdout` に
     持つ。

各段階で、native と全ターゲットの golden を通す。境界の外の形は、最も早い
静的な境界で拒否する。
