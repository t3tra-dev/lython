> Note: Please do not send pull requests to this repository.

# pyc - Python to LLVM IR transpiler & compiler

```
              🚀 Benchmark Results              
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ runtime        ┃ time              ┃ result  ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ LLVM           │ 25.20ms (x1.63)   │ 9227465 │
│ C              │ 36.26ms (x2.34)   │ 9227465 │
│ Python(pyc)    │ 44.66ms (x2.89)   │ 9227465 │
│ Bun            │ 55.38ms (x3.58)   │ 9227465 │
│ Deno           │ 81.69ms (x5.28)   │ 9227465 │
│ Node.js        │ 98.11ms (x6.34)   │ 9227465 │
│ Python         │ 640.84ms (x41.41) │ 9227465 │
│ Python(no GIL) │ 928.87ms (x60.02) │ 9227465 │
└────────────────┴───────────────────┴─────────┘
```

**pyc** は、Python コードを LLVM IR に変換 (トランスパイル) し、さらに機械語へコンパイルすることを目指した実験的プロジェクトです。  
CPython とは異なる形で静的型付けのように扱いながらPythonソースを解析し、`clang` などのツールチェーンを利用してネイティブバイナリを生成することを目標としています。

## Features
- Python AST をトラバースし、LLVM IR を生成
- 生成された IR をさらに `clang` や `llc` などでコンパイルして実行ファイル化を目指す
- Python によるソースコード解析部分は静的型に近い形で扱う実験的実装
- ランタイム (`runtime/`) として最低限のメモリ管理・`print` 関数などを C 実装で提供

---

## Directory Structures

```text
├── .gitignore
├── .python-version        # Python のバージョン指定 (3.12)
├── .vscode
│   └── settings.json      # VSCode 用設定ファイル
├── bench.py               # ベンチマーク用スクリプト
├── benchmark/             # ベンチマークで使用するコード群 (C/JS/LLVM IR/Pythonなど)
│   ├── cfib.c
│   ├── jsfib.js
│   ├── llfib.ll
│   ├── pyfib.py
│   └── pyfib.py.ll
├── helloworld.ll          # サンプルの "Hello, world!" LLVM IR
├── pyc/
│   ├── __init__.py
│   ├── __main__.py        # `python -m pyc` で呼ばれるエントリーポイント
│   ├── codegen/           # Python→LLVM IR 変換ロジック
│   │   ├── ir/            # LLVM IR を構築するためのビルダー等
│   │   └── visitors/      # 各種 AST ノードへの Visitor 実装
│   └── compiler/          # 生成された LLVM IR をバイナリに変換するロジック (ll2bin など)
├── pyproject.toml         # Python プロジェクト管理用 (PEP 621)
├── runtime/
│   ├── runtime.c
│   └── runtime.h          # C 言語で実装したランタイム (メモリ管理やprint関数など)
├── sample.c               # C のサンプルコード
├── sample.ll              # 上記 C コードを LLVM IR 化した例
├── source.py              # Python のサンプルコード
├── source.py.ll           # source.py を LLVM IR 化した例
├── source.py.s            # さらにアセンブリまで生成した例
└── uv.lock                # uv による依存関係のロックファイル
```

---

## インストール

### 環境構築

このリポジトリでは [uv](https://docs.astral.sh/uv) というパッケージ管理ツールを使用しています。  
Python 3.12 以上が必要です (`.python-version` で 3.12 を指定しています)。

1. **uv のインストール**  
   Unix:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Windows:
   ```bash
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **依存関係の同期**  
   ```bash
   uv sync
   ```
   `uv.lock` の内容に従って各種パッケージ (black, isort, rich など) をインストールします。

3. **コンパイラツールチェーン**  
   LLVM/Clang がインストールされている必要があります。  
   `clang --version` や `llc --version` が使用できる状態にしてください (環境に応じてインストール)。

---

## 使用方法

### LLVM IR の生成

```bash
python -m pyc --emit-llvm <python_file>
```

例: `source.py` を LLVM IR 化して `source.py.ll` を出力する:
```bash
python -m pyc --emit-llvm source.py
```
実行後、同じフォルダに `source.py.ll` が生成されます。

### バイナリへのコンパイル

```bash
python -m pyc --compile <python_file>
```

例: `source.py` を機械語バイナリにコンパイルする:
```bash
python -m pyc --compile source.py
```
実行後、(現状は実験的ですが) `source.py.out` が生成される想定です。

### AST のダンプ

```bash
python -m pyc --dump-ast <python_file>
```
Python の AST (抽象構文木) を文字列としてダンプします。  
内部的には `ast.dump()` 相当の機能を使用しています。

---

## ベンチマーク

`bench.py` を使うと、以下のような言語/コンパイルパターンでフィボナッチ (n=35) の実行時間を比較できます:

- Node.js / Bun / Deno (JavaScript)
- C (clang) with -O0, -O1, -O2, -O3
- LLVM IR (clang) with -O0, -O1, -O2, -O3
- Python (pyc で LLVM IR 化したバイナリ)
- Python (CPython), Python(no GIL) (3.13 スレッドフリー版など)

#### 使い方

```bash
python bench.py
```

スクリプトの冒頭 `setup()` 関数で C や LLVM IR のコンパイルを行い、その後各エグゼを繰り返し呼び出して平均実行時間を比較します。  
結果はターミナルに表形式で表示されます (内部で `rich` を使用)。

---

## ランタイム

`runtime/` ディレクトリ以下に、C 言語で実装された最小限のランタイムが含まれています。

- `runtime.c` / `runtime.h`
  - `create_string`, `free_string`, `int2str`, `str2str`, `print` などの関数を提供
  - LLVM IR 上で `declare` してコールすることで、Python の組み込み風の機能を実装

このランタイムをリンクして最終的なバイナリを生成することにより、`print("Hello")` などが動作します。

---

## 今後の予定 / 注意点

- **型推論の強化**: 現在は非常に簡易的に int/str などを仮定しているのみ。関数呼び出しにおける引数・戻り値の型チェックなどは未実装に近いです。
- **制御構文の拡張**: `if` 以外の制御構文 (`while`, `for`, `try` など) はまだ多くが未実装。
- **クラス・例外対応**: クラス定義や例外処理など、Python の主要機能のほとんどは未対応。
- **最適化パス**: 生成した LLVM IR をどのように最適化するかは今後の課題です。
- **Windows など他プラットフォーム対応**: 開発環境は Unix 系 (Linux, macOS) を想定しています。Windows での動作確認は限定的です。

本プロジェクトはあくまで実験的段階のため、上記のように不完全な部分や将来的に大きな変更が入る可能性があります。

---

## ライセンス

本リポジトリのソースコードは、特記がない限り [MIT License](https://opensource.org/licenses/MIT) で配布されています。  
詳細はソースコード内の記述（`pyc/__init__.py` など）をご参照ください。
