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

## 未実装 (段階 2 以降)

- コールバック (Python の callable を JS 関数として渡す)。寿命は
  `FinalizationRegistry` で管理する。境界をまたぐ循環参照はリークする
  (Pyodide と同じ)。
- `await` による Promise の待機。WebLoop 方式を採る。待つ間は wasm から JS に
  戻り、Promise の解決で再開する。JSPI はランタイムの対応待ち。ただし、
  サスペンドを抽象の後ろに置き、JSPI のバックエンドを足せる形にしておく。
- `isinstance(x, js.Element)` による絞り込み (JS の `instanceof`)。
- `to_js` / `to_py` (list、dict、TypedDict の変換)。
- グローバルへの代入と、union 型のグローバルの読み出し。今は lowering が
  拒否する。
- DOM を使うテスト (playground の Playwright を使う)。
