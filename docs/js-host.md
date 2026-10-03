# JavaScript ホストとの境界 (`js` モジュール)

Pyodide の `from js import ...` に相当する機能の設計メモ。対象はブラウザ向けの
`wasm32-unknown-emscripten` / `wasm64-unknown-emscripten`。他のターゲットで
`import js` すると、ドライバが「JavaScript ホストのないターゲット」として拒否する。

## 型: スタブが契約

`js` の型は `src/lython/runtime/lib/js.pyi` (約 3.3 万行)。TypeScript の
`lib.es*.d.ts` / `lib.dom.d.ts` を pyodide-playground の
`scripts/js-stubgen.mjs` が Python に起こしたもので、生成物をそのまま置いている
(冒頭に再生成の手順)。

- `.pyi` は一般に「実装のない契約」として扱う。スタブのクラスは、フィールド、
  メソッド、`@overload` を protocol table (`py::protocols::Table`) に登録し
  (`declareStubClassContracts`)、ユーザーのクラスと同じ仕組みでメンバーを解決する。
- スタブの `@overload` は PEP 484 のとおり宣言順に試し、最初に受理したものを選ぶ
  (`ProtocolMethod::firstApplicable`)。manifest の overload は従来どおり点数で
  選び、同点は曖昧として拒否する。
- `t.Literal["div"]` の引数は、実引数を広げる前に照合する
  (`createElement("div")` の overload を選べるようにするため)。
- `js` には `StubContractPolicy` を適用する。
  - 戻り値の `t.Any` は `_js.JsProxy` (属性を持たない JS 値) にする。
  - 引数の `t.Any` は `object` にする。
  - `@staticmethod` は、クラスの値を受け手とするメソッドとしても登録する。
    JS の static は、コンストラクタを `this` にして呼ぶものだから。
- `from js import X` は Pyodide と同じく `globalThis.X` を指す。スタブの
  モジュールレベルの変数と、`Window` のプロパティ (`Math`, `JSON`,
  `URLSearchParams` …) がグローバルになり、同名のクラスより優先する。
  `X.new(...)` は `new X(...)` を表す。
- スタブのエイリアス (`type BodyInit = ...`) は契約を組み立てる間だけ束縛し、
  プログラムの名前空間には出さない。
- グローバルと同名のクラスは、注釈の中でだけ効くエイリアスとして束縛する
  (`def f(p: URLSearchParams)`, `"js.Object"`)。値として読めばグローバル。
- スタブのクラスはすべて `_js.JsProxy` を基底に持つ。JS 値かどうかの判定
  (`isJsHostType`) はこの関係で行う。そのため、プログラム自身の `js.py` の
  クラスとは混同しない。
- `isinstance(x, C)` で C がホストのコンストラクタのとき (`new` の結果が T)、
  T に対する JS の `instanceof` を実行時に行い、真の側で x を T に絞り込む
  (`IsInstanceAnalysis::Kind::HostTest`)。Lython の class id の比較には決して
  回さない (JS 値の class id はすべて同じなので、答えにならない)。union の値は
  先に None などを外す必要があり、そうでなければ拒否する。

## 表現: 1 つの runtime 契約

型検査はスタブの精密な型 (`js.Document` など) で行う。そのうえで、Phase 8c
(`eraseJsHostContracts`) が module 中のすべての `js.*` を `_js.JsProxy`
(`runtime/modules/_js.mlir`) に置き換える。以降の lowering が扱う JS 値の型は
これ 1 つだけになる。メンバーへのアクセスに必要な情報は、名前と、両端の静的な型
(引数の型と結果の型) だけで、どちらも op に残っている。

- `_js.JsProxy` は `[refcount, class id, handle]` という形で、幅は 17
  (`HandleWidthRegistry.h`)。解放するとホスト側の handle も落とす。
- `_js.mlir` は `ly.runtime.only_with = "ly.js.host"` を持つ。`js` を import
  したプログラムにだけ取り込まれる。
- 値は、静的な型に従って境界を越える。
  - Python から JS へ: None は undefined、bool は boolean、int は number
    (double で正確に表せる範囲) または BigInt (それを超える範囲。i64 を超える
    値は 10 進表現経由)、float は number、str は string (code point の幅のまま
    渡すので、孤立サロゲートも保たれる)、JsProxy はその値自身になる。
  - JS から Python へ: スタブの宣言どおりの型を JS 側で検査する。食い違えば
    TypeError にする (スタブは読んだ所で検査し、鵜呑みにしない)。
  - union 型の結果 (`Element | None` など): emitter
    (`adaptJsHostResult`) が、None、スカラー、JS 値の順に種類を判定し、
    分岐と合流で union を組み立てる。所有権の検証器が辿れる形にするため。
- JS 側の例外は RuntimeError (`"<名前>: <メッセージ>"`) にする。Pyodide の
  `JsException` は独立したクラスとして未実装。`except JsException` は、名前が
  未定義なのでコンパイル時に拒否される。

## ホスト側: グルーの本体とアダプタ

- 本体は `src/lython/runtime/js/lython_js.js`。Emscripten を知らず、wasm の
  メモリは渡されたアクセサからだけ読む。lyc に埋め込まれている。
- 引数は値ごとに `LyJs_Push*` で積み、呼び出し側がまとめて受け取る。配列を
  wasm メモリに並べる必要も、一時的な handle も不要になる。
- Emscripten 用のアダプタは、lyc がリンクのたびに、オブジェクトが実際に
  import している `LyJs_*` から `--js-library` として生成する。本体は
  `--pre-js` で渡す。jsifier は、`--js-library` に定義のない import を、
  モジュール名に関係なく拒否する。そのため、wasm の独自 import モジュールは
  使えない。
- **Emscripten を剥がすとき**に差し替えるのは、このアダプタの 2 ファイル分
  だけになる。本体を `lython_js` などの import モジュールとして渡す自前の
  loader を書けばよい。

## WASI のローダ (`--js-host`)

`lyc --target wasm32-wasip1 --js-host -o prog.js` は、Emscripten を使わずに
`js` を持つプログラムを作る。出力は `prog.wasm` とローダ `prog.js` の 2 つ。

- ローダは `lython_js.js` (本体はそのまま) と `runtime/js/lython_wasi.js`
  (WASI preview-1 の shim と起動) から成る。`node prog.js` でも、ブラウザの
  `<script>` でも動く。
- プログラムは、ホストの関数を `lython_js` から import し、`LyJs_Dispatch` /
  `LyJs_Release` を export する (LLVM の段で付ける属性による)。
- shim が答えるのは、ファイルシステムを持たないプログラムが使う呼び出しだけ。
  preopen がないので、ファイルを開くと ENOENT になる。
- JSPI がある所では、`time.sleep` と、asyncio がホストを待つ点で、プログラムを
  中断する。その間も JS のイベントループが進む (docs/async-design.md の
  段階 7)。
- `sys._js_host` は、Emscripten と `--js-host` 付きの WASI で True。

## コールバック

スタブが callback を受け取ると宣言している引数に Python の callable (関数、
クロージャ、bound method) を渡すと、JS 側には呼び出すとそれを実行する JS 関数が
渡る。

- 受け渡しの側ごとに、emitter が包み関数 `__ly_js_wrap$K` を合成する。K は、
  宣言された callback の型と callable 自身の型の組で決まる。包み関数は、利用者の
  callable を引数なしのクロージャで包み、ブリッジ (`runtime/lib/_js_bridge.py`)
  に登録して、その slot の JS 関数を得る。
- クロージャの中では、JS の引数を `js.$arg$<i>$<K>` というホストグローバル
  として読む。型は宣言された引数の型で、変換は他のホストの値と同じ。結果は、
  内部用のホストのクラス `js.$CallbackFrame` の `set_result` で返す。
  callable が値を返すなら、宣言が `Any` でも返す (JSON の reviver の結果は、JS が
  保持する値だから)。`$` はプログラムが書けない名前なので、衝突しない。
- JS から呼ばれる入口は、引数も戻り値もない Python 関数 `_js_bridge.dispatch`
  1 つだけ。lyc が LLVM の段で C の入口 `LyJs_Dispatch` (解放は
  `LyJs_Release`) を作り、Emscripten から export する。
- 例外は `dispatch` が捕まえる。JS の関数は、`"<型>: <メッセージ>"` を持つ
  Error を投げる。それがホストの呼び出しから戻ると、Python 側では RuntimeError
  になる。
- 寿命: JS の関数が回収されたら、`FinalizationRegistry` が slot を外す。
  回収は JS のイベントループで走るので、同期的なループの中で作ったコールバックは
  main が終わるまで残る (実測で 1 個あたり約 1 KB)。境界をまたぐ循環参照は
  リークする (Pyodide と同じ)。
- 制限:
  - lambda の引数には、宣言された型が伝わらない (注釈が必要)。
  - コールバックを渡すプログラムは、main が返ってもランタイムを終えない
    (`-sEXIT_RUNTIME=0`)。`setTimeout` などで後から呼ばれる。
  - 属性に callable を代入する形 (`el.onclick = f`) は未対応で、lowering が
    拒否する。

## 未実装 (段階 3 以降)

- `await` による Promise の待機は実装済み (docs/async-design.md の段階 6)。
  asyncio のループがホストのループで進む (WebLoop) ので、待つ間は wasm から
  JS に戻る。`asyncio.run()` のようにブロックしたまま Promise を待つことは、
  Emscripten では RuntimeError になる。WASI のローダ (下記) では JSPI で
  中断して待つ。
- `to_js` / `to_py` (list、dict、TypedDict の変換)。
- グローバルへの代入と、union 型のグローバルの読み出し。今は lowering が
  拒否する。
- DOM を使うテスト (playground の Playwright を使う)。
