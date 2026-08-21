# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

Lython は LLVM/MLIR 22 ベースの Python コンパイラツールチェーン。Python ソースを独自の py dialect (MLIR) に変換し、LLVM IR まで下げて JIT 実行または AOT コンパイルする。C++17 / CMake + Ninja。LLVM/MLIR はメジャーバージョン 22 に固定されており、CMake が version mismatch を FATAL_ERROR で拒否する。

**プロジェクトの性格**: Python を静的型付けコンパイル言語として再実装するプロジェクトである。基本的に静的に解決できるコードのみを受け入れ、`object` / `Any` に対するランタイム操作は実装しない (動的ディスパッチへのフォールバックは存在せず、静的に解決できない構造は最も早い静的境界で診断を出して拒否する — "never silently mis-execute")。

**安全性設計**: メモリ安全証明を根冠とする。根拠は (1) Hindley-Milner 準拠の型推論 — Algorithm J 基盤 (union-find + occurs check、`emitter/lib/TypeInference.h`) と Algorithm M 的な期待型伝播の複合で、manifest 駆動 subtype/protocol は subsumption レイヤとして共存する。facade は `TypeSystem` (`emitter/lib/TypeSystem.h`)、詳細は `docs/type-inference.md`。(2) 量的型理論 (QTT) に基づく参照カウントの挿入・最適化 — `Own(rho)` を affine capability として扱い、quantitative ownership verifier が挿入・elision・検証を通して同一の alias 関係を共有する。証明カーネルの定義は `rfc/memory-safety-proof.md` (実装の詳細はこの judgment を具体化する場合にのみ有効)。

## ビルド・実行コマンド

```bash
uv sync                       # Python 依存 (pyright のみ)
cmake -B build -S .           # LLVM/MLIR 22 を Homebrew / apt パスから自動検出
cmake --build build -j$(nproc)

# JIT 実行
./build/bin/lyc jit examples/hello.py

# AOT コンパイル
./build/bin/lyc examples/fib.py -o fib_aot

# AST ダンプ (parse サブコマンド)
./build/bin/lyc parse examples/hello.py

# 型チェック (src/lyrt のスタブに対して strict モード)
uv run pyright
```

主な `lyc` フラグ: `--emit-llvm` (LLVM IR で停止)、`--release` (verifier パス無効化)、`--target` / `-mcpu` / `-mfpu` (クロスコンパイル)、`--fsanitize=...`、`--audit-runtime-manifest`、`-jit-codegen-opt=none|less|default|aggressive` (JIT の命令選択品質。デフォルト none は初回出力レイテンシ優先、計算律速なら aggressive で AOT 相当)、`-mmatrix=auto|sme|amx|none` (行列エンジン選択。auto は公開 ISA の SME 優先、amx は Apple AMX を runtime probe 付きで強制)。

### テスト

ctest スイート (`LYTHON_BUILD_TESTS`, デフォルト ON)。CI (`.github/workflows/ci.yml`) も `ctest` を実行する:

```bash
ctest --test-dir build -j8 --output-on-failure   # 全件 (447 件、99 s / RelWithDebInfo)
ctest --test-dir build -L fast                   # lowering を通らない層 (100 件、1.0 s)
ctest --test-dir build -L emit                   # emitter を触ったとき
ctest --test-dir build -LE bench                 # 実行律速のベンチを除く
ctest --test-dir build -N -L fast                # 何が選ばれるか (実行しない)
```

**ラベルは「テストが到達するパイプライン段」で切ってある。ファイル名やディレクトリでは切らない。**
`tests/unit/` には 0.05 s の emit テストと 17 s の matmul フルコンパイルが同居し、`golden.errors` 109 件は emit で 0.09 s、lowering 中 0.20 s、実行が必要な 60 件に分かれる。「unit = 安い」「golden = 高い」はどちらも成り立たない (`rfc/test-suite-debt.md`)。

- 段ラベル (1 テスト 1 つ): `parse` / `emit` / `tables` / `meta` / `lower` / `e2e`
- コストラベル: `fast` (lowering を通らない) / `slow` (単体 20 s 超・コンパイル律速) / `bench` (単体 20 s 超・実行律速)
- スイートラベル: `unit` / `golden` / `golden-cases` / `golden-errors` / `examples`

`slow` と `bench` を分けてあるのは、前者は lowering の改善で縮み後者は縮まないから (`examples.tarai` は 96.9% が実行なので lowering 改善の上限は 3%)。混ぜると修正が効いたかが見えなくなる。**ラベルの妥当性は機械が検査する**: unit 側は `LayerManifestTest` がどの層にも属さないテストと何も掴まないパターンを名前付きで落とし、golden 側は `tests/golden/layers.txt` の宣言を `run_case.py --expect-layer` がコンパイラ自身の PerfScope 痕跡と照合する。

- **ユニットテスト** (`tests/unit/`, GoogleTest) — C++ API を直接叩く。ドライバ API (`src/lython/driver/include/Driver.h`) は lyc・テスト・fuzzer が共有するコンパイル入口。
- **golden テスト** (`tests/golden/`) — `cases/*.py` は stdout 完全一致、`errors/*.py` は exit code + stderr 正規表現。実行前に止まるケースは `layers.txt` に段を宣言する。
- **examples smoke** — golden とバイト同一でないものだけ exit code 検証。同一な 20 件は golden がより強い assertion を持つので登録しない (`golden.example_twins` が同一性を守る)。

#### 新しいテストを追加するときの判断

**上から順に、最初に該当した層に置く。下の層で落とせるものを golden に書かない。** 同じプログラムで emit のみ 0.05 s、フルコンパイル 1.7 s。

1. **parse だけで決まるか** → `ParserTests.cpp`。
2. **emitter の診断か** (名前解決 / 型解決 / 未対応構文の拒否) → `emitMLIRFromSource` で診断文字列を assert (`EmitterTests.cpp`)。**golden を書かない。**
3. **verifier / lowering の拒否か** → `compilePythonSourceToLLVMIR` の失敗を assert (`DriverTests.cpp`)。jit-build と実行を払わない。
4. **stdout / traceback / 実行時の値が必要か** → そのときだけ golden。**「なぜ実行が必要か」を 1 行書く。書けないなら 3 で足りる。**
5. **実行前に止まる golden は `layers.txt` に段を宣言する。** `ctest -L fast` に入り、段がずれたら赤になる。
6. **unit テストを足したら層に登録する。** どの層にも入っていないテストはどこでも走らない。`LayerManifestTest` が落として教えるが、落ちてから直すより先に書くほうが安い。

PR に書くこと:

- **この経路を覆う既存のテストは何か。無いことをどう確かめたか。** 名前や見た目の類似は根拠にしない — 83,028 ペアで同一は 20 組、SUBSET は 1 組だけだった。「似ているから重複」は 3 回外れている。
- **その golden が赤になれる根拠。** 修正と同時に書く golden は修正前のバイナリで赤になることを確認する (`tests/probe/tools/redcheck.py --sentinel`)。していないならそう書く。
- **20 s を超えるテストを足したら `LYTHON_PERF=1` のフェーズ内訳を貼る。** それはテスト設計ではなくコンパイラの欠陥の報告である可能性が高い。そして `LYTHON_SLOW_TESTS` / `LYTHON_BENCH_TESTS` のどちらかを**測って**決める (ビルド種別を跨いで比較しない)。

やらないこと:

- **`examples/` に golden の twin を置かない。** サンプルは `examples/` にだけ置く。
- **`.stderr-re` に `^` アンカーを増やさない。** stderr に書くデバッグ機能で静かに壊れる (現存 11 件)。
- **cwd にファイルを書く golden を増やさない。** 同名を足した瞬間に `-j8` で非決定的になる。
- **新しい重量フェーズを sub-`PerfScope` なしで追加しない。** 計装の穴はコストの穴と同じ場所に開く。

`slow` / `bench` の所属は手動であり、上の PR チェックリストだけが強制する — **スイートは新たに遅くなったテストを自力では検出できない。**

### Fuzzing (libFuzzer)

`fuzz/` に 3 つの harness: `fuzz_parser` (parse のみ)、`fuzz_emitter` (parse→emit)、`fuzz_pipeline` (フルパイプライン、JIT 実行なし)。ツリー全体を ASan + カバレッジ計装するため専用ビルドディレクトリが必要 (macOS は Homebrew LLVM clang 必須、AppleClang 不可):

```bash
cmake -B build-fuzz -S . -G Ninja \
  -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
  -DLYTHON_ENABLE_FUZZERS=ON -DLYTHON_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0:allow_user_poisoning=0 cmake --build build-fuzz -j$(nproc)
ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0:allow_user_poisoning=0 \
  ./build-fuzz/bin/fuzz_pipeline -max_total_time=300 -timeout=10 -rss_limit_mb=4096 \
    -dict=fuzz/dictionaries/python.dict fuzz/corpora/fuzz_pipeline tests/golden/cases
```

`ctest --test-dir build-fuzz` はチェックイン済みコーパス + golden cases の無変異リプレイ (regression)。テストは harness ごとに `corpus` (チェックイン済みコーパス、crash regression) と `goldens.1..4` (golden の `.py` を 4 分割) に分かれる — 1 プロセスで全部 replay すると ASan 下で 13 分 (CI ではその 2.5 倍) かかり、golden が増えるほど伸びるため。`-j` で並列に走らせること。発見した crash 入力は診断化して修正後、該当 `fuzz/corpora/<harness>/` に追加する。

### デバッグ用環境変数

- `LYTHON_IR_DUMP=all` またはフェーズ名のカンマ区切り (例: `frontend,runtime-lowering`) — 各 lowering フェーズ後の MLIR / LLVM IR を stderr にダンプ
- `LYTHON_PERF=1` — フェーズごとの wall time を出力
- `LYTHON_NUM_THREADS=N` — 実行時: 大きい行列積の fork-join ワーカー数 (デフォルト 4、1 で逐次)

## アーキテクチャ

コンパイルパイプライン: **parser → C++ AST → emitter (py dialect MLIR) → lowering pipeline → LLVM dialect → LLVM IR → ORC JIT / AOT リンク**。ドライバは `tools/CLI.cpp` (単一ファイルの `lyc`)。

- `src/lython/parser/` — CPython 3.14 の PEG parser (`parser.c`, `python.gram`, `Python.asdl`) をベンダリングし、CPython ランタイムなしでリンクできるようパッチしたもの。生成パーサが受理した後に C++ AST builder が公開 `Node` ツリーを作る二段構え。改変の詳細と制約は `src/lython/parser/CPYTHON_PATCHES.md` 必読。
- `src/lython/dialects/` — py dialect の TableGen 定義 (`PyDialect.td`) と型・プロトコル実装。生成ヘッダに依存するターゲットは `PyDialectIncGen` への明示的依存が必要 (並列ビルドの race 対策)。
- `src/lython/emitter/` — AST → py dialect の emit と型システム (`TypeSystem.cpp`, `PrimitiveTypes.cpp`)。
- `src/lython/lowering/Passes/LoweringPipeline.cpp` — **フェーズ順序の single source of truth**。番号付きフェーズが verifier → 最適化 → runtime import → manifest strategy 適用 → runtime lowering → refcount 挿入/elision → LLVM 変換の順に走る。
- `src/lython/runtime/` — 標準モジュール群の実装。`lib/` と `modules/` はそれぞれ **CPython の `Lib/` と `Modules/` に対応**する: `lib/*.py` は pure-Python で記述できる標準モジュール、`modules/*.mlir` は CPython が C で実装しているようにネイティブ実装が必要な標準モジュール (contract + ネイティブ実装 + transform dialect による lowering strategy を 1 ファイルに同居させたマニフェスト)。いずれも **typeshed に基づく型契約** (`ly.typing.*` 属性 / `.pyi` 準拠の annotation) と **CPython 本流に基づく実装** (well-typed で静的コンパイル可能な表面に制限した port) を持つ。CPython からの逸脱はファイル冒頭 docstring に明記する慣行 (`collections.py` 参照)。**`lib/*.py` は全 def の完全注釈を規約とする** — 未注釈パラメータ推論はメインモジュールの pre-pass 限定で import 経路には走らないうえ、typeshed 契約との一致検証は明示注釈が前提。`lib/*.py` はさらに二種類: runtime-internal モジュール (`stackguard_support.py` など、`LythonRuntimePyLowering` でターゲット triple ごとに MLIR bytecode へ事前 lowering され全プログラムにリンク、import 不可) と、stdlib モジュール (それ以外全部、ソースのまま `lyc` に埋め込まれ import 解決される)。埋め込み処理は `tools/CMakeLists.txt` にある。
- `src/lython/verifier/` — フェーズ間をゲートする verifier 群 (evidence / native / affine-ownership)。
- `src/lyrt/` — lyrt (Lython runtime library) の型スタブのみ (`.pyi`)。pyright strict の対象。

### Lowering の設計ルール (docs/lowering-architecture.md 必読)

新しい変換をどこに置くかはこのドキュメントの層構造に従う:

- **フェーズ間で必ず走る変換** → `LoweringPipeline.cpp` のフェーズ。
- **py op の lowering** → `RuntimeBundleLowerer` の `lower*` メソッド (`lowering/Passes/Runtime/Core/Dispatch.cpp` の TypeSwitch でディスパッチ)。この層は意図的に `DialectConversion` を**使わない**: lowering が `RuntimeBundle` (`Runtime/Model/Bundles.h`) に蓄積される SSA 値ごとの evidence に依存するため。
- **物理型マッピング** → `runtimeValueTypesFor` の隣 (`Runtime/ABI/`)。
- **bundle 状態不要の stateless rewrite** → py dialect の canonicalization パターン。
- **モジュール固有・スケジュール的な変換** → `runtime/modules/*.mlir` 内の `__lython_strategy_*` transform named sequence (C++ に書かない)。

プロジェクトの基本原則: **never silently mis-execute** — 未対応の構造は最も早い静的境界で明示的な診断を出して拒否する。legality の保証は `ConversionTarget` ではなくフェーズ間 verifier が担う。

### ctypes コールバックの signal-safety ポリシー

`ctypes.CFUNCTYPE` コールバック本体は allocation-free でなければならない (`Ly*` ランタイム呼び出し・malloc 系到達禁止)。最終 LLVM dialect IR に対して `Passes/Runtime/Ctypes/CallbackThunks.cpp` の `verifyCallbackSignalSafety` が強制する。詳細は `docs/lowering-architecture.md` 参照。

## ドキュメンテーション規約

- **コード**には How を書く。
- **テストコード**には What を書く。
- **コミットログ**には Why を書く。
- **コードコメント**には Why NOT を書く (なぜ別のやり方をしなかったか。How の説明や変更の正当化はコメントに書かない)。
