> Please do not send pull requests to this repository.

# Lython - Python compiler toolchain based on LLVM

> [!TIP]
> Searching for **pyc**? You are in the right repo. **pyc** has been renamed to **Lython**.

```
              🚀 Benchmark Results              
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ runtime        ┃ time              ┃ result   ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ LLVM(O1)       │ 15.71ms (x0.69)   │ 9227465  │
│ C(O1)          │ 16.37ms (x0.72)   │ 9227465  │
│ LLVM(O3)       │ 17.17ms (x0.76)   │ 9227465  │
│ C(O3)          │ 17.30ms (x0.76)   │ 9227465  │
│ C(O2)          │ 17.88ms (x0.79)   │ 9227465  │
│ LLVM(O2)       │ 20.32ms (x0.90)   │ 9227465  │
│ Lython         │ 22.68ms (x1.00)   │ 9227465  │
│ LLVM(O0)       │ 23.15ms (x1.02)   │ 9227465  │
│ C(O0)          │ 33.34ms (x1.47)   │ 9227465  │
│ Bun            │ 54.52ms (x2.40)   │ 9227465  │
│ Deno           │ 80.33ms (x3.54)   │ 9227465  │
│ Node.js        │ 94.68ms (x4.17)   │ 9227465  │
│ Python         │ 611.73ms (x26.97) │ 9227465  │
│ Python(no GIL) │ 884.05ms (x38.97) │ 9227465  │
└────────────────┴───────────────────┴──────────┘
```

**Lython** は、Python コードを LLVM IR に変換 (トランスパイル) し、機械語へコンパイルすることを目指した実験的プロジェクトです。 
LLVM を基盤にしつつ、CPython とは異なる形で静的型付けのように扱いながら Python ソースを解析し、`clang` などのツールチェーンでネイティブバイナリを生成することをゴールとしています。

## Features
- **Python AST のトラバース**: Python の抽象構文木を解析して LLVM IR を生成
- **ツールチェーン活用**: 生成した IR をさらに `clang` や `llc` などでコンパイルし、実行ファイル化を狙う
- **実験的な静的型解析**: Python によるソースコード解析部分を簡易的に静的型チェック風に処理
- **ランタイム (`runtime/`) の自前実装**: メモリ管理 (Boehm GC) や `print` 関数などを最低限 C 言語で提供

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
├── runtime/                   # C で実装したランタイム (Boehm GC)
│   └── builtin/
│       ├── functions.c
│       ├── functions.h
│       ├── types.c
│       └── types.h
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

`runtime/` ディレクトリ配下に C言語で実装した最小限のランタイム（Boehm GC 使用）が置かれています。

- `builtin/functions.c / .h`  
  - `PyInt_FromI32`, `PyList_New`, `int2str`, `print` などを提供  
  - LLVM IR から `declare` 呼び出しすることで、Python 組み込み風の関数を実装
- `builtin/types.c / .h`  
  - `String`, `PyInt`, `PyList` などの型を定義  
  - 動的メモリ管理は Boehm GC に任せ、Python 的なオブジェクトを簡易的に再現

これらのランタイムをリンクすることで、`print("Hello, world!")` やリスト操作などの処理が可能になります。

---

## 今後の予定 / 注意点

- **型推論の強化**: まだ `int` / `str` など一部型にしか対応していない
- **制御構文の拡張**: `while`, `for`, `try` やクラス定義は未実装
- **最適化パス**: ほぼ `clang -O2` などに丸投げ。将来的に LLVM の最適化パスをカスタムする可能性あり
- **Windows 等のサポート**: 開発は主に Unix 系 (Linux, macOS) を想定。Windows での検証は限定的です

本プロジェクトは実験的段階のため、今後仕様変更が入る場合があります。  
興味を持っていただけた方は、連載ブログ記事やサンプルを参考に、ぜひ試してみてください！

---

## ライセンス

本リポジトリのソースコードは、特記がない限り [MIT License](https://opensource.org/licenses/MIT) で配布されています。  
詳細はソースコード内の記述（`lython/__init__.py` など）をご参照ください。
