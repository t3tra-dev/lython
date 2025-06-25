## Lython - トップレベル構造案

> **狙い**
>
> - "静的型付け + 高速 AOT" を核に、**Python 互換文法** と **厳格な型システム** を両立
> - **MLIR ベース** で中間表現を多段階管理 (独自 _Python Dialect_ -> LLVM Dialect)
> - **PyObject 系譜** と **ネイティブ系譜** の共存を前提に拡張しやすいレイヤリング
> - ビルド/テスト/ドキュメント/ツール群を CI まで一体化し、長期保守可能な構造

---

### 1. リポジトリ直下

```
lython/
├─ README.md           ... プロジェクト概要
├─ docs/               ... Sphinx/Tutorial/ADR
├─ scripts/            ... 開発用ユーティリティ
├─ cmake/ or build/    ... CMakeLists/Bazel WORKSPACE など
├─ pyproject.toml      ... Python ラッパ CLI 用 (hatch)
├─ third_party/
└─ src/
```

- **ビルド系** は _CMake + LLVM ExternalProject_ を推奨

  - MLIR/LLVM をサブモジュール化せず外部ビルドで pin
  - Python パッケージ面は `pyproject.toml` で wheel 化し CLI を配布

---

### 2. `src/` 以下のコアレイヤ

```
src/
├─ lython/
│  ├─ frontend/        <-  Python 解析 & TypedAST
│  │   ├─ parser/      ... CPython AST ラッパ
│  │   ├─ typing/      ... 型推論エンジン
│  │   └─ passes/      ... Syntax & Type check
│  ├─ dialects/
│  │   ├─ python/      ... MLIR Python Dialect (ODS + C++/Py bindings)
│  │   ├─ runtime/     ... 実行時 ABI 用 dialect (ref, gc.write など)
│  │   └─ native/      ... ネイティブ高速演算 dialect (optional)
│  ├─ ir/              ... 共通 IR builder API (Facade)
│  │   └─ mlir_builder.py
│  ├─ passes/          ... MLIR Pass Library
│  │   ├─ lowering/
│  │   ├─ optim/
│  │   └─ codegen/
│  ├─ backends/
│  │   ├─ llvm/        ... obj/asm 生成
│  │   └─ jit/         ... 実験的 JIT (optional)
│  ├─ runtime/         ... C/C++ ランタイム & GC
│  └─ cli/             ... lythonc, litest (テストランナー)
└─ tests/
```

| レイヤ       | 主責務                                       | 拡張ポイント                                   |
| ------------ | -------------------------------------------- | ---------------------------------------------- |
| **frontend** | CPython AST → TypedAST 生成・型検査          | 型推論アルゴリズム/プラグイン (mypy-plugin 風) |
| **dialects** | MLIR 方言定義 (ODS + TableGen)               | Python Dialect の拡張、別言語方言の追加        |
| **ir**       | “ビルダー抽象層” — 生成先を MLIR/LLVM に切替 | 新バックエンド実装時もフロント側変更最小化     |
| **passes**   | Dialect 間 Lowering・最適化                  | パスを Python/C++ 両方で実装可 (mlir-python)   |
| **backends** | LLVM CodeGen・JIT・WASM …                    | 将来 GPU/CPP/Swift バックエンドも追加可能      |
| **runtime**  | PyObject 実装・GC・FFI Shim                  | GC 実装交換、`native` ABI 拡張                 |

---

### 3. TypedAST サブシステム

| コンポーネント                   | 説明                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------ |
| `frontend/parser/ast_builder.py` | `ast` モジュールを走査し _Node_ を `TypedNode` にラップ                        |
| `frontend/typing/solver.py`      | Hindley–Milner + gradual constraints。関数内ローカル推論 -> グローバル注釈伝播 |
| `frontend/typing/types.py`       | `PrimitiveType(i32/i64/f64)`, `PyObjectType`, `Generic[T]`, `Any`, `Union`     |
| `frontend/passes/type_check.py`  | 型不一致をエラー収集。`native` 部分は Strict，一般 Python 部分は Widen         |
| 拡張                             | `plugin/` で **Pydantic** 等独自型ルールを注入可                               |

---

### 4. MLIR Python Dialect

```
dialects/python/
├─ PythonOps.td      ... ODS 定義
├─ PythonTypes.td
└─ Py.cpp / PyDialect.cpp
```

- **運用方針**

  - 可能な限り高位演算 (`Py.Add`, `Py.Call`, `Py.MakeList`) で保持
  - 型推論済みなので `result` Type は `!py.i32` / `!py.obj` 等で確定
  - `lowering/` パスで `scf`, `linalg`, `memref` 等へ分割
  - JIT 実行用に `py.exec` テストランナー (`lit`) を同梱

---

### 5. `native` API の ABI ルール

| 要素                       | 実装イメージ                                                                              |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `native.i32`, `native.f64` | Dialect Level 型 `!native.i32` -> 下位で `i32`                                            |
| `@native` デコレータ       | Frontend で属性付与 -> Dialect op `native.func`                                           |
| 変換                       | `to_native`, `to_pytype` -> Dialect op `py.cast` 系                                       |
| ABI                        | C シンボルとして `extern "C"` でエクスポート、FFI 用ヘッダ自動生成 (`scripts/gen_abi.py`) |

---

### 6. GC 層

1. **Phase 1**: CPython 互換参照カウントのみ (循環はテストで抑止)
2. **Phase 2**: LLVM _shadow-stack_ GC + `llvm.gcroot` (精密マーク)
3. **Phase 3**: Statepoint & Concurrent GC (任意)

GC を差し替えられるように `runtime/gc/iface.h` を抽象化し、`runtime/gc/rc/`, `runtime/gc/shadow/` など実装フォルダを切替ビルド。

---

### 7. DevOps & CI

- **CI**: GitHub Actions で

  - _Linux/macOS/Windows_ 向けビルド
  - `lit` で MLIR パス単体テスト
  - `pytest` でフロントエンド/型推論ユニットテスト
  - `bench/` ベンチを `asv` や `pytest-benchmark` で測定

- **フォーマット**: `clang-format`, `black`, `isort`
- **ドキュメント**: `docs/adr/` にアーキテクチャ決定記録、`docs/dev_guide/` にハック手順

---

### 8. 初期ブートストラップ (90 日目標)

| 週    | マイルストーン                                         |
| ----- | ------------------------------------------------------ |
| 1–2   | ビルドシステム雛形/LLVM + MLIR external build          |
| 3–4   | Frontend: CPython AST 取り込み + TypedNode stub        |
| 5–6   | 最小型推論 (`int`, `str`, `bool`) & TypeError 検出     |
| 7–8   | Python Dialect ODS 定義・`py.func`, `py.constant`      |
| 9–10  | MLIR Builder Facade (`mlir_builder.py`) + IR 生成      |
| 11–12 | Lowering: `py.constant` -> `arith.constant`            |
| 13    | LLVM Dialect 変換 & `llc` で "Hello, World" コンパイル |
| 14–15 | `native` API MVP (`@native` + i32) 動作確認            |
| 16–18 | 基本演算と if/while -> `scf` Lowering, CI 緑化         |

---

### 9. 将来拡張のための規約

| 規約                   | 目的                                                        |
| ---------------------- | ----------------------------------------------------------- |
| **Stable IR Boundary** | `dialects/python` までを"公開 Front-IR"として互換を守る     |
| **Pass Plugin API**    | `passes/` 配下は `REGISTER_PASS` マクロで DSO プラグイン可  |
| **Runtime ABI Ver.**   | `LY_ABI_VERSION` マクロを導入、バイナリ互換を明示           |
| **Feature Gate**       | `-fpy-long`, `-fnative-simd` など機能フラグで実験機能を切替 |

---

### 10. チーム構成（例）

| ロール             | 担当範囲                         |
| ------------------ | -------------------------------- |
| **Language Front** | Parser / TypedAST / type checker |
| **Dialect & IR**   | MLIR 方言定義・Pass・Builder     |
| **Backend**        | LLVM CodeGen・ランタイム・GC     |
| **Dev Infra**      | Build / CI / Docs / Release      |
| **Product Lead**   | ロードマップ管理・仕様凍結       |

---

この構造を土台にすれば、**静的型付け x Python 互換** のコアを保ちつつ、MLIR ベースで GPU・WASM・別言語 runtime など多方面へ拡張可能です。
