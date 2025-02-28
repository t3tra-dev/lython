> Please do not send pull requests to this repository.

# Lython - Python compiler toolchain based on LLVM

> [!TIP]
> Searching for **pyc**? You are in the right repo. **pyc** has been renamed to **Lython**.

```
              🚀 Benchmark Results               
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ runtime        ┃ time              ┃ result   ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ LLVM(O1)       │ 15.64ms (x0.58)   │ 9227465  │
│ C(O1)          │ 15.89ms (x0.59)   │ 9227465  │
│ LLVM(O3)       │ 18.00ms (x0.67)   │ 9227465  │
│ C(O2)          │ 18.31ms (x0.68)   │ 9227465  │
│ C(O3)          │ 18.51ms (x0.69)   │ 9227465  │
│ LLVM(O2)       │ 20.59ms (x0.77)   │ 9227465  │
│ LLVM(O0)       │ 24.35ms (x0.91)   │ 9227465  │
│ Lython         │ 26.75ms (x1.00)   │ 9227465  │
│ C(O0)          │ 33.93ms (x1.27)   │ 9227465  │
│ Bun            │ 52.09ms (x1.95)   │ 9227465  │
│ Deno           │ 75.02ms (x2.80)   │ 9227465  │
│ Node.js        │ 93.08ms (x3.48)   │ 9227465  │
│ Python         │ 623.53ms (x23.31) │ 9227465  │
│ Python(no GIL) │ 887.40ms (x33.18) │ 9227465  │
└────────────────┴───────────────────┴──────────┘
```

**Lython** は、Python コードを LLVM IR に変換 (トランスパイル) し、機械語へコンパイルすることを目指した実験的プロジェクトです。 
LLVM を基盤にしつつ、CPython とは異なる形で静的型付けのように扱いながら Python ソースを解析し、`clang` などのツールチェーンでネイティブバイナリを生成することをゴールとしています。

## Features
- **Python AST のトラバース**: Python の抽象構文木を解析して LLVM IR を生成
- **ツールチェーン活用**: 生成した IR をさらに `clang` や `llc` などでコンパイルし、実行ファイル化を狙う
- **実験的な静的型解析**: Python によるソースコード解析部分を簡易的に静的型チェック風に処理
- **CPython互換オブジェクトシステム**: CPythonのオブジェクトシステムに準拠したランタイム実装
- **参照カウント方式のメモリ管理**: Boehm GCを基盤としつつ、CPython互換の参照カウント方式を実装

---

## Directory Structures

```text
├── .gitignore
├── .python-version            # Python のバージョン指定 (3.12)
├── .vscode/                   # VSCode 用設定ファイル
│   ├── settings.json
│   └── c_cpp_properties.json
├── bench.py                   # ベンチマーク用スクリプト
├── benchmark/                 # ベンチマークで使用するコード群 (C/JS/LLVM IR/Pythonなど)
│   ├── cfib.c
│   ├── jsfib.js
│   ├── llfib.ll
│   └── pyfib.py
├── helloworld.ll              # サンプルの "Hello, world!" LLVM IR
├── src                        # メインソース
│   ├── lython
│   │   ├── __init__.py
│   │   ├── codegen
│   │   │   ├── ir/            # LLVM IR を構築するためのビルダー等
│   │   │   └── visitors/      # 各種 AST ノードへの Visitor 実装
│   │   └── compiler/          # 生成された LLVM IR をバイナリに変換するロジック (ll2bin など)
│   └── lythonc
│       └── __main__.py        # CLIのエントリポイント
├── pyproject.toml             # Python プロジェクト管理用 (PEP 621)
├── runtime/                   # CPython互換のランタイム実装
│   └── builtin/
│       ├── functions.c        # 基本的な組み込み関数の実装
│       ├── functions.h
│       ├── macro_exports.c    # マクロのエクスポート
│       └── objects/           # オブジェクトシステムの実装
├── samples/                   # 他言語から生成したIRのサンプル
├── Makefile                   # ランタイムのビルド用
├── source.py                  # Python のサンプルコード
├── source.py.ll               # source.py を LLVM IR 化した例
├── source.py.s                # さらにアセンブリまで生成した例
└── uv.lock                    # uv による依存関係のロックファイル
```

---

## インストール

### 環境構築

このリポジトリでは [uv](https://docs.astral.sh/uv) というパッケージ管理ツールを使用しています。  
Python 3.12 以上が必要です (`.python-version` で 3.12 を指定しています)。

1. **uv のインストール**
   - **Unix (Linux/macOS)**:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - **Windows**:
     ```powershell
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
2. **依存関係の同期**
   ```bash
   uv sync
   ```
   これにより `uv.lock` の内容に従い、black, isort, rich などをまとめてインストールします。
3. **コンパイラツールチェーン**  
   - LLVM/Clang が必要です。`clang --version` や `llc --version` が使用できる状態にしておきましょう。
4. **ランタイムのビルド**  
   - ルートディレクトリにある `Makefile` を使い、`runtime.o` を生成します:
     ```bash
     make
     ```
   - `make clean` でキャッシュやバイナリを削除可能です。

---

## 使用方法

### 1. LLVM IR の生成

```bash
python -m lythonc -emit-llvm <input-path>
```
- 例:  
  ```bash
  python -m lythonc -emit-llvm source.py
  ```
  実行後、`source.py.ll` が同一ディレクトリに生成されます。

### 2. バイナリへのコンパイル

```bash
python -m lythonc <input-path> -o <output-path>
```
- 例:  
  ```bash
  python -m lythonc source.py -o main
  ```
  実行後、 `main` バイナリが生成されます。

---

## ベンチマーク

`bench.py` を使うと、以下のような言語/ランタイム/コンパイルパターンでフィボナッチ (n=35) の実行時間を比較できます:

- Node.js / Bun / Deno (JavaScript)
- C (clang) with -O0, -O1, -O2, -O3
- LLVM IR (clang) with -O0, -O1, -O2, -O3
- **Lython**
- Python (CPython), Python(no GIL)

#### 使い方

```bash
python bench.py
```

- スクリプト内の `setup()` 関数で事前に C や LLVM IR のコンパイルを行い、その後それぞれ実行して平均実行時間を測定します。  
- 結果はターミナル上に表形式（`rich`）で出力されます。

---

## ランタイム

`runtime/` ディレクトリ配下に C言語で実装したCPython互換のオブジェクトシステムが置かれています。

### オブジェクトシステム

- **基本設計**: CPythonのオブジェクトシステムに準拠した設計
- **参照カウント方式**: Boehm GCを基盤としつつ、`Py_INCREF`/`Py_DECREF`による参照カウント管理
- **型オブジェクト**: `PyTypeObject`を中心とした型システムの実装

### 組み込み関数

- `runtime/builtin/functions.c` で基本的な組み込み関数を実装
- `print` などの基本的な関数をサポート
- LLVM IRからこれらの関数を呼び出すことで、Python風の動作を実現

---

## 今後の予定 / 注意点

- **型推論の強化**: まだ `int` / `str` など一部型にしか対応していない
- **制御構文の拡張**: `while`, `for`, `try` やクラス定義は未実装
- **最適化パス**: ほぼ `clang -O2` などに丸投げ。将来的に LLVM の最適化パスをカスタムする可能性あり
- **Windows 等のサポート**: 開発は主に Unix 系 (Linux, macOS) を想定。Windows での検証は限定的です

本プロジェクトは実験的段階のため、今後仕様変更が入る場合があります。  
興味を持っていただけた方は、[Zennの記事](https://zenn.dev/t3tra/articles/056b406cb688da)やサンプルを参考に、ぜひ試してみてください！

---

## ライセンス

本リポジトリのソースコードは、特記がない限り [MIT License](https://opensource.org/licenses/MIT) で配布されています。  
詳細はソースコード内の記述（`lython/__init__.py` など）をご参照ください。
