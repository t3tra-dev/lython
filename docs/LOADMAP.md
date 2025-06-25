# Lython Rebuild Roadmap (2025‒2027)

## 0. Vision & Guiding Principles

- **目的**: Python 文法互換でありながら _TypeScript レベルの厳格な静的型検査_ と _AOT コンパイルによる C/C++ 並み性能_ を実現する。
- **柱**: ① **TypedAST** による厳密な型解析、② **PyObject 系譜** と **Native 系譜** の両立、③ **MLIR ベース** の多層 IR パイプライン、④ **LLVM Shadow‑Stack GC** を基盤とした軽量ランタイム。
- **アウトプット**: lythonc (CLI), liblython (ライブラリ), VS Code 拡張, 標準ランタイム (~50 KB) など。

---

## 1. 技術基盤

| レイヤ           | 採用技術                              | 備考                                   |
| ---------------- | ------------------------------------- | -------------------------------------- |
| フロントエンド   | **Python 3.13 `ast` + 自作 TypedAST** | Python 公式 AST をラップし型属性を付与 |
| 中間表現  (自作) | **Python Dialect (MLIR)**             | AST と 1:1 互換。静的型注釈を保持      |
| 中間表現  (高位) | **scf / tensor / linalg Dialect**     | 制御フロー・テンソル演算を表現         |
| 中間表現  (低位) | **memref / arith Dialect**            | バッファ化後の数値演算                 |
| 最終 IR          | **LLVM Dialect -> LLVM IR 20+**       | AOT オブジェクト生成                   |
| ランタイム       | **Shadow-Stack GC (LLVM GC)**         | 参照カウント併用を検討                 |

---

## 2. マイルストーン概要

| フェーズ                                | 期間 (月) | 主成果物                                        | 主要課題                  |
| --------------------------------------- | --------- | ----------------------------------------------- | ------------------------- |
| **Phase 0 – Kick-off**                  | 0‑1       | Monorepo, CI/CD, clang‑format, docs 雛形        | 開発ワークフロー確立      |
| **Phase 1 – TypedAST Core**             | 1‑4       | パーサ／型推論エンジン, 1 万行テスト            | typing PEP604, ユニオン型 |
| **Phase 2 – Type System v1**            | 3‑6       | PyObject/Native 型統合 API, `native` デコレータ | 暗黙/明示変換ルール策定   |
| **Phase 3 – Python Dialect**            | 4‑8       | MLIR Python Dialect v0.1, IR printer/parser     | Dialect 定義 & TableGen   |
| **Phase 4 – High‑Level Passes**         | 7‑10      | 型検証 Pass, scf/linalg 変換                    | SSA 変換 & DomTree        |
| **Phase 5 – Lowering to LLVM**          | 9‑12      | memref -> LLVM Dialect Lowering                 | ABI/calling conv.         |
| **Phase 6 – Runtime & GC**              | 10‑14     | Shadow‑Stack GC PoC, librt.a                    | write barrier 実装        |
| **Phase 7 – AOT Toolchain**             | 12‑16     | `lythonc` CLI, object/link                      | クロスコンパイル          |
| **Phase 8 – Optimization & Benchmarks** | 14‑18     | O3 + LTO + 型特化最適化                         | 速度 > CPython x50        |
| **Phase 9 – Tooling**                   | 15‑20     | LSP, VS Code Ext, デバッガ                      | Source-map / DWARF        |
| **Phase 10 – RC & Docs**                | 18‑22     | v0.9 β, チュートリアル, API Doc                 | 互換性保証                |
| **Phase 11 – 1.0 Release**              | 22‑24     | v1.0  タグ, pkg managers                        | SemVer 適用               |
| **Phase 12 – Post‑1.0**                 | 24‑36     | GPU Dialect, auto‑vectorize, JIT mode           | SPIR-V backend            |

---

## 3. フェーズ別詳細タスク

### Phase 0 Kick‑off (Month 0‑1)

- **リポジトリ初期化**: GitHub -> `lython`。MIT License。
- **CI/CD**: GitHub Actions で Linux/macOS/Windows ビルド。clang20+MLIR/LLVM20。
- **Dev Env**: nix/devcontainer 提供。clang‑format 14, mypy, pytest。
- **RFC  フロー**: 仕様変更は pull request + docs/rfcs/ に提案。

### Phase 1 TypedAST Core (Month 1‑4)

1. **AST ラッパ生成**: `TypedNode` 基底に `.ty` 属性 (PythonType) を追加。
2. **一次型推論**: 変数初期化 & 関数シグネチャから型を決定。PEP484/PEP563 に準拠。
3. **制御フロー解析**: SSA 変換用のブロック情報を付加。
4. **型エラー報告**: TypeScript 風のエラー (expected vs. actual) を CLI に整形出力。
5. **ユニットテスト**: `pytest` で 1  万行カバレッジ 80 %以上。

### Phase 2 Type System v1 (Month 3‑6)

- **型クラス実装**: `PrimitiveType`, `ObjectType`, `GenericType`, `UnionType`, `OptionalType`。
- **`native` API v0**: `@native`, `i32/i64/f32/f64`, `to_native`, `to_pytype`。
- **双方向変換**: コンパイル時に自動／手動変換を生成。Implicit/explicit 変換規則を docs に整理。
- **アノテーションインポート**: `from __future__ import annotations` 互換。`typing` モジュールの `typing.List[int]` 等を解析。

### Phase 3 Python Dialect (MLIR) (Month 4‑8)

1. **Dialect 定義**:

- 型: `!py.int`, `!py.float`, `!py.obj`, `!py.i32` 等。
- Ops: `py.call`, `py.add`, `py.build_list`, `py.box`, `py.unbox`。

2. **Printer/Parser**: MLIR TableGen で自動生成。
3. **Lowering Pass v0**: `py.*` -> `scf`, `tensor` 等へ変換 (ネイティブ計算は `arith.*`)。
4. **IR Verification**: Dialect の TypeVerifier で静的整合性チェック。

### Phase 4 High‑Level Passes (Month 7‑10)

- **型検証 Pass**: TypedAST -> Python Dialect 間で保持した型が scf/tensor 上でも崩れないか検証。
- **オブジェクト <-> ネイティブ境界挿入**: `py.box/unbox` の自動挿入アルゴリズム。
- **データフロー最適化**: SSA + copy‑prop, dead‑code‐elim 。
- **分割コンパイル**: モジュール単位のパスパイプライン。

### Phase 5 Lowering to LLVM (Month 9‑12)

1. **memref -> LLVM Dialect**: Bufferization + pointer arith。
2. **ランタイム ABI**: `PyObject*` レイアウト、C symbol 連携 (`extern "C"`)。
3. **CodeGen**: `mlir::translateModuleToLLVMIR` -> `llc`。
4. **クロスコンパイル**: x86‑64, aarch64 Linux/macOS/Win。

### Phase 6 Runtime & GC (Month 10‑14)

- **Shadow‑Stack GC**: `gc "shadow-stack"` を関数属性に付与し `llvm.gcroot` を生成。
- **参照カウント v0**: PyObject ベース型には RC を残す (後に削除検討)。
- **アロケータ**: `ly_alloc`, `ly_free` (mimalloc 互換 API 上、tagged ptr 検討)。
- **ランタイム API**: list/dict/str 最小実装; fmt, hash, equal.

### Phase 7 AOT Toolchain (Month 12‑16)

- **CLI**: `lythonc <file.py> -o a.out`。clang と同等の UX。
- **Linker**: lld をデフォルト使用。 `--gc-sections` 有効。
- **パッケージング**: wheel 生成, `pip install lython` で lythonc とライブラリを配置。

### Phase 8 Optimization & Benchmarks (Month 14‑18)

- **LLVM O3 + LTO**: ThinLTO, PGO サポート。
- **型特化**: オブジェクト型に対して _guard + devirtualize_ パス。
- **micro‑benchmarks**: PyPI 上位 50 パッケージのコア関数をテスト。

### Phase 9 Tooling (Month 15‑20)

- **LSP**: hover/type‑info/diagnostics/rename。
- **VS Code Ext**: syntax highlight, jump‑to‑def, inline IR viewer。
- **デバッガ**: dwarf line info -> lldb 連携、MLIR viewer。

### Phase 10 RC & Docs (Month 18‑22)

- **ドキュメントサイト**: mkdocs + material theme。
- **チュートリアル**: "Lython tutorial", 30  章。
- **API 安定化**: `typing` 準拠と互換性ガイド。

### Phase 11 1.0 Release (Month 22‑24)

- CI  バッジ  100%, CodeCov 90% 以上。
- semantic versioning & GitHub Releases。
- Homebrew/Nix/Chocolatey formulas。

### Phase 12 Post‑1.0 (Month 24‑36)

- **GPU backend**: MLIR -> SPIR-V Dialect -> AMD/NVIDIA。
- **Auto‑vectorize**: VPlan / MLIR Vector Dialect。
- **JIT**: `mlir::ExecutionEngine` + ORC JIT による REPL モード。

---

## 4. 横断的取り組み

- **QA/CI**: clang‐address‐sanitizer, UBSan, Valgrind。
- **セキュリティ**: fuzzing (libFuzzer) によるフロントエンド強化。
- **ガバナンス**: 3  か月ごとに RFC レビュー会, ロードマップ更新。
- **リスク & 緩和**: MLIR API 変更 -> サブツリー Embed; GC 実装難 -> 段階的移行; 人員不足 -> タスク自動計測。

---

## 5. 推奨チーム構成 (最小)

| 役割             | 人数 | ミッション                              |
| ---------------- | ---- | --------------------------------------- |
| Compiler Lead    | 1    | 全体設計, 仕様決定, パイプライン統括    |
| Frontend & Type  | 2    | Lexer/Parser, TypedAST, typing PEP 対応 |
| MLIR Architect   | 2    | Dialect 定義, Pass 設計, Lowering       |
| Runtime Engineer | 2    | PyObject 基盤, Shadow‑Stack GC          |
| Tooling & Infra  | 1    | CLI, LSP, CI/CD                         |
| QA/Perf          | 1    | Benchmark, Profiling, Fuzzing           |

---

## 6. 初期 90  日アクションアイテム

1. **Day 0‑7**: リポジトリ・CI 立ち上げ, コーディング規約策定。
2. **Day 7‑14**: TypedAST スケルトン, E2E  ハローワールド通過 (int 加算)。
3. **Day 14‑30**: 型推論パス v0, 100  テストケース。
4. **Day 30‑60**: `native` API MVP, MLIR Dialect 雛形。
5. **Day 60‑90**: scf/linalg 変換パス, LLVM Lowering PoC, `sum.py` ベンチ 50x 速達成。

---

### 付録 A – 参考仕様・ドキュメント

- MLIR Python Bindings Slide (2024 LLVM Dev Mtg) \[Zinenko]。
- MLIR Docs: linalg, scf, tensor, memref Dialect。
- LLVM GC Shadow‑Stack ドキュメント。
- Kaleidoscope Chapter GC 例。
